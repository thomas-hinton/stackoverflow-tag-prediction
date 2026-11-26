# The frozen split is iteratively stratified on Tags

The train/validation/test split (70/15/15, per Question) uses multi-label iterative stratification (Sechidis, Tsoumakas and Vlahavas, 2011) rather than a plain seeded shuffle. The rarest Tags are placed first, and each of their Questions goes to the set that still lacks the most of that Tag; set sizes are fixed up front from the ratios by the largest remainder method. On the current dataset every one of the 86 Supported tags gets at least 14 training, 3 validation and 3 test Questions, which is exactly what ADR 0003 needs for per-Tag scores and macro-F1 to mean something. The implementation is about 40 lines in `so_tagger.split`, draws every random choice from one seeded generator, and so reproduces the committed files byte for byte.

Because the split is committed and both Taggers are compared on it, changing the algorithm later would invalidate earlier results; that is why it is recorded here.

## Considered Options

- **Seeded random split**: rejected, because a Tag carried by exactly 20 Questions then lands at most once in the 15% test split about 18% of the time (ADR 0003), leaving its per-Tag score meaningless.
- **A library implementation** (`iterative-stratification` / scikit-multilearn): rejected, because it adds a dependency for a short algorithm, and its output across library versions is not something we control, while the frozen split must stay reproducible.
