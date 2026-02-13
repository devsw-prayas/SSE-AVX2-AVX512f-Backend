# SSE-AVX2-AVX512f-Backend
Standard Vectorization Headers that my projects use

# Generating Project-Specific Headers

This repository contains identity-neutral SIMD headers.

To integrate them into a project (e.g. StormSTL or Leibniz), use the GitHub Action to generate project-bound versions.

---

## How to Generate Headers

1. Go to the **Actions** tab.
2. Select **"Generate SIMD Artifact"**.
3. Click **Run workflow**.
4. Fill in the required fields:

### Inputs

* **namespace**
  The namespace to inject.
  Example: `StormSTL`

* **forceinline**
  The FORCEINLINE macro used by your project.
  Example: `STORM_FORCEINLINE`

* **nodiscard**
  The NODISCARD macro used by your project.
  Example: `STORM_NODISCARD_MSG`

5. Click **Run workflow**.

---

## Downloading the Result

After the workflow completes:

1. Open the workflow run.
2. Download the generated artifact ZIP.
3. Extract the headers.
4. Place them inside your project.

---

## What Gets Generated

The following files are rewritten with your identifiers:

```
v128Intrin.h
v256Intrin.h
v512Intrin.h
```

All namespace and macro tokens are replaced deterministically.

The canonical source in this repository is never modified.

---

## Important

* Always regenerate headers after updating this repository.
* Do not manually edit generated headers.
* This repository remains the single source of truth.

---

If you want an even leaner version (like ultra-minimal 10-line README), I can compress it further.

