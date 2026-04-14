# Отчёт: REINFORCE-поиск бинарных путей в диффузии

## 1. Общая идея

Задача: найти оптимальную бинарную маску длиной `N_BITS = 14`, которая на каждом шаге диффузии выбирает, какой промпт использовать — **source** или **target**. В итоге получается изображение, у которого фон максимально похож на source, а передний план — на target.

Полный перебор: `2^14 = 16384` масок, каждая требует полного прогона диффузии (~5с). Это ~22 часов на одной A100.

REINFORCE учит **распределение Бернулли** над каждым битом и находит оптимум примерно за `~1500-2000` изображений — в **8–10× быстрее** полного перебора.

---

## 2. Политика

```python
class BernoulliPolicy:
    def __init__(self, n_bits=14, init_logit=0.0):
        # 14 независимых обучаемых logit-параметров (нет нейросети!)
        self.logits = nn.Parameter(torch.full((n_bits,), init_logit))

    def sample(self, batch_size):
        probs = sigmoid(self.logits)              # (14,)
        dist = Bernoulli(probs)
        masks = dist.sample((batch_size,))        # (B, 14)
        log_probs = dist.log_prob(masks).sum(-1)  # (B,) суммируем по битам
        return masks, log_probs
```

**Важно**: это не нейросеть, а просто 14 скаляров. `init_logit=0` означает `P(target) = 0.5` для каждого бита — максимальная энтропия на старте.

---

## 3. Функция награды

Для каждого сгенерированного изображения `img` считается:

### 3.1 `bg_ssim` — сохранение фона

```python
# SSIM между img и source_image, но усреднённое только по пикселям фона
smap = ssim_map(img, source_img, gaussian_kernel)  # (B, 1, H, W)
bg_ssim = (smap * bg_mask).sum() / n_bg_pixels     # в диапазоне [0, 1]
```

`bg_mask` получается из CLIPSeg по seg_prompt (например, `"cat"`) — 1 где фон, 0 где объект.

### 3.2 `fg_clip` — соответствие переднего плана target-промпту

```python
# 1. Обрезаем квадратный bbox вокруг переднего плана
fg_crop = img[:, :, y1:y2, x1:x2]

# 2. Прогоняем через SigLIP 2 SO400M (vision encoder)
fg_emb = F.normalize(siglip2.get_image_features(fg_crop))  # (B, 1152)

# 3. Предвычисленная "delta" направление в текстовом пространстве
delta_text = F.normalize(tgt_text_emb - src_text_emb)      # (1152,)

# 4. Проекция fg на это направление
fg_clip = (fg_emb * delta_text).sum(-1)                    # (B,) в ~[-0.3, 0.3]
```

Эта формула отвечает на вопрос: «насколько передний план сдвинут от source к target в пространстве CLIP?»

### 3.3 Финальная награда

```python
reward = alpha * bg_ssim + (1 - alpha) * fg_clip
# alpha = 0.3 (по умолчанию): больше вес на передний план
```

**Без клампинга** — отрицательный `fg_clip` штрафует, не обнуляется.

---

## 4. Цикл обучения (REINFORCE)

```python
policy = BernoulliPolicy(n_bits=14)
optimizer = Adam([policy.logits], lr=0.1)
baseline = 0.0                                    # EMA базовой линии
best_reward = -inf; best_mask = None
episodes_since_improvement = 0

for ep in range(num_episodes):                    # num_episodes = 400
    # --- 1. Sample batch of masks ---
    masks, log_probs = policy.sample(batch_size)  # (8, 14), (8,)

    # --- 2. Generate images (no grad, diffusion is black-box) ---
    with torch.no_grad():
        images = flux_generator.generate(masks)   # (8, 3, 512, 512)
        rewards, bg, fg = reward_computer.compute_rewards(images, alpha=0.3)

    # --- 3. Update EMA baseline ---
    baseline = 0.9 * baseline + 0.1 * rewards.mean()

    # --- 4. Advantage standardization ---
    advantage = rewards - baseline
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # --- 5. REINFORCE loss ---
    policy_loss = -(advantage.detach() * log_probs).mean()
    entropy = policy.entropy()                    # сумма энтропий по 14 битам
    loss = policy_loss - entropy_coeff * entropy  # entropy_coeff = 0.05

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --- 6. Track best-ever mask from exploration ---
    if rewards.max() > best_reward:
        best_reward = rewards.max()
        best_mask = masks[rewards.argmax()]
        episodes_since_improvement = 0
    elif ep >= min_episodes:                      # ← FIX: счётчик после min_episodes
        episodes_since_improvement += 1

    # --- 7. Early stopping (after min_episodes floor) ---
    if ep >= min_episodes:
        if entropy < entropy_stop:                # 0.5 → политика коллапсировала
            break
        if episodes_since_improvement >= plateau_patience:  # 150 → плато
            break
```

### Ключевые моменты

1. **Градиенты НЕ идут через диффузию** — `generator.generate(masks)` в `torch.no_grad()`. Диффузия — чёрный ящик. Градиент течёт только через `log_probs` от политики. Это "score function estimator" REINFORCE.

2. **Бейзлайн**: EMA среднего награды, снижает дисперсию градиента. Без него REINFORCE на бинарных задачах слишком шумный.

3. **Advantage normalization**: `(A - mean)/(std + eps)` внутри батча, стандартизирует шкалу градиента между эпизодами.

4. **Entropy bonus**: `−entropy_coeff * H(π)` добавляется к loss (знак минус, чтобы максимизировать энтропию). Предотвращает раннюю деградацию политики к 50/50 или полному коллапсу.

5. **Лучшая маска от exploration** (`best_mask`) хранится отдельно от выученной политики. Это важно — REINFORCE часто находит хорошую маску случайным сэмплированием ещё до того, как политика полностью сойдётся. После обучения используется `best_mask`, а не `greedy = (probs > 0.5)`.

---

## 5. Ранняя остановка (early stopping)

Три условия, все обрабатываются только после `ep >= min_episodes = 200`:

| Условие | Логика | Типичный триггер |
|---|---|---|
| **Hard cap** | `ep >= num_episodes (400)` | редко достигается |
| **Entropy floor** | `H(π) < 0.5` | политика почти детерминистична |
| **Plateau** | нет улучшения `best_reward` за 150 эпизодов после floor | награда вышла на плато |

### Почему min_episodes floor важен

Раньше был баг: счётчик `episodes_since_improvement` считался с эпизода 0. Если лучшая награда находилась на эпизоде 5 и потом не улучшалась, к эпизоду 200 счётчик был уже 195, и plateau-stop срабатывал немедленно. Результат: 8 из 16 запусков останавливались ровно на 201-м эпизоде.

**Фикс** (в `generation/reinforce_search.py`):
```python
elif ep >= args.min_episodes:
    episodes_since_improvement += 1
```
Теперь счётчик копится только после floor, давая политике реальное окно для исследования.

---

## 6. Генерация изображений (диффузия)

```python
def generate(self, masks):  # masks: (B, 14)
    # Расширяем 14 бит в 28 шагов диффузии (каждый бит повторяется 2 раза)
    step_masks = masks.repeat_interleave(2, dim=1)  # (B, 28)

    latents = shared_noise.expand(B, -1, -1).clone()
    for i, t in enumerate(timesteps):              # 28 шагов
        mask_i = step_masks[:, i]                  # (B,) bool
        # На каждом шаге выбираем src или tgt эмбеддинг
        pe = torch.where(mask_i, tgt_embeds, src_embeds)

        # FLUX transformer predict noise, twice (conditional + unconditional для CFG)
        noise_cond = transformer(latents, t, encoder_hidden_states=pe)
        noise_uncond = transformer(latents, t, encoder_hidden_states=neg_embeds)
        noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(noise, t, latents)

    images = vae.decode(latents)  # (B, 3, 512, 512)
    return images
```

**Shared noise** — одна и та же начальная нойза для всех масок (одинаковый `seed`). Это единственный способ честно сравнивать маски — без этого случайность инициализации доминировала бы над эффектом маски.

---

## 7. Финальная фаза (Phase 5): топ-K изображений

```python
# 1. Greedy mask — порог 0.5 на выученных вероятностях
greedy_mask = (policy.probs > 0.5).unsqueeze(0)

# 2. K-1 дополнительных масок из выученного распределения
extra_masks, _ = policy.sample(top_k - 1)

# 3. Также best-ever маска из exploration
all_masks = cat([best_mask, greedy_mask, extra_masks])

# Генерируем все, сортируем по награде, сохраняем top-K jpg
```

Это даёт разнообразие в финальных результатах: и оптимум политики, и лучшее из случайных сэмплов.

---

## 8. Гиперпараметры (по умолчанию)

| Параметр | Значение | Обоснование |
|---|---|---|
| `n_bits` | 14 | Половина от 28 диффузионных шагов FLUX |
| `batch_size` | 8 | Компромисс между шумом градиента и throughput |
| `num_episodes` | **400** | Hard cap, редко достигается |
| `min_episodes` | **200** | Floor перед любой ранней остановкой |
| `plateau_patience` | **150** | 150 эпизодов без улучшения → стоп |
| `entropy_stop` | 0.5 | Коллапс политики → стоп |
| `lr` | 0.1 | Adam для 14-мерных логитов |
| `alpha` | 0.3 | 30% на bg_ssim, 70% на fg_clip |
| `entropy_coeff` | 0.05 | Поощрение exploration |
| `baseline_ema` | 0.9 | Медленный EMA базовой линии |
| `vision_model` | SigLIP 2 SO400M @384px | Замена CLIP ViT-B/32 — в 2-3× сильнее сигнал |

---

## 9. Что reward отвечает математически

Матожидание награды по политике:

$$J(\theta) = \mathbb{E}_{m \sim \pi_\theta}\left[R(m)\right] = \sum_{m \in \{0,1\}^{14}} \pi_\theta(m) \cdot R(m)$$

где `π_θ(m) = ∏_i Bern(sigmoid(θ_i))` — факторизованное Бернулли-распределение.

REINFORCE теорема:

$$\nabla_\theta J(\theta) = \mathbb{E}_{m \sim \pi_\theta}\left[R(m) \cdot \nabla_\theta \log \pi_\theta(m)\right]$$

В коде это превращается в:

$$\text{loss} = -\frac{1}{B}\sum_b \underbrace{(R_b - b)}_{\text{advantage}} \cdot \log \pi_\theta(m_b) - c \cdot H(\pi_\theta)$$

где `b` — EMA бейзлайн, `c = 0.05` — коэффициент энтропии. Знак минус: PyTorch минимизирует, а мы хотим максимизировать `J`.

---

## 10. Что работает хорошо и что нет

**Работает:**
- Поиск оптимальной маски за ~1500-2000 изображений (8-10× reduction)
- Политика находит универсальный паттерн: первые биты `b0, b1` → source (сохраняют структуру), средние `b2-b8` → target (редактирование)
- SigLIP 2 даёт достаточно сильный сигнал, чтобы даже «похожие» промпт-пары (sunflower→lavender) не коллапсировали

**Проблемы:**
- Greedy mask `(probs > 0.5)` не всегда совпадает с `best_mask` — политика не полностью коммитится. Поэтому мы используем `best_mask` как финальный результат.
- Hard случаи (bg-rich с 10+ объектами на фоне) требуют больше эпизодов — обычно застревают с entropy > 2.0.
- Градиенты пропорциональны абсолютной шкале награды, а она зависит от конкретной пары промптов. Нет способа перенести гиперпараметры между задачами без тюнинга.

---

## Файлы

- `generation/reinforce_search.py` — основной файл, классы `BernoulliPolicy`, `DiffusionGenerator`, `RewardComputer`, функция `train_reinforce`
- `analysis/reinforce_summary.py` — сводная статистика, heatmap вероятностей, сетки top-K
- `analysis/reinforce_insights.py` — анализ convergence speed, bit correlations, sample efficiency
- `scripts/rerun_all_v3.sh` — запуск всех 16 экспериментов в 2 волнах
- `scripts/rerun_8_with_fix.sh` — валидационный запуск 8 экспериментов на одной волне
