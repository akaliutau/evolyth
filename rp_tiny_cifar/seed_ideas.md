# Seed mutation ideas for the first generation

1. **Narrow depthwise CNN**
   - Reduce channels to 16/24/48.
   - Keep residual depthwise blocks.
   - Expected: fewer params and lower latency.

2. **Wider shallow CNN**
   - Use fewer blocks but wider early layers.
   - Expected: faster training, maybe better early accuracy.

3. **Squeeze-excitation lite**
   - Add tiny channel attention only in 48/64 channel blocks.
   - Expected: better accuracy for small parameter cost.

4. **ConvMixer-lite**
   - Patch embed with stride 2, then repeated depthwise + pointwise mixer blocks.
   - Expected: visually interesting architecture card.

5. **Low-rank classifier head**
   - Replace final linear with 64 -> 24 -> 10.
   - Expected: modest parameter reduction.

6. **Anti-overfit tiny model**
   - Smaller network, stronger dropout, maybe label smoothing by editing only model behavior if needed.
   - Expected: robust on 5k training subset.
