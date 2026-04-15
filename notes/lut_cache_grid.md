# LUT x cache-size x generation-length grid

Solo benchmarks of the F.1 accumulating-cache ANE DFlash draft across quantization, cache size (state_length S), and generation length (max_new). Target: `mlx-community/Qwen3-4B-bf16` on GPU. Draft: `z-lab/Qwen3-4B-DFlash-b16` ported to ANE (100% ANE placement).

Hardware: Mac mini M4 Pro, 64 GB. Prompts (n=4): capital, fibonacci, math, story. Values = mean tok/s across prompts.

## Unquantized fp16

| state_length \ max_new | 100 | 300 | 1000 |
|---:|---:|---:|---:|
| S=512 | 17.51 | 8.62 | 9.97 |
| S=1024 | 28.69 | 16.06 | 12.00 |
| S=2048 | 24.05 | 13.32 | 21.90 |
| S=4096 | 21.47 | 16.39 | 18.29 |

<details><summary>Per-prompt detail</summary>

**S=512, max_new=100** — mean 17.51 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 12.12 | 8.25 | 56 | 99 | 14.33 |
| fibonacci | 100 | 2.70 | 37.04 | 14 | 99 | 12.93 |
| math | 100 | 6.34 | 15.77 | 28 | 99 | 15.18 |
| story | 100 | 11.13 | 8.99 | 52 | 99 | 14.10 |

**S=512, max_new=300** — mean 8.62 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 34.02 | 8.82 | 157 | 299 | 14.32 |
| fibonacci | 300 | 29.16 | 10.29 | 105 | 299 | 14.71 |
| math | 300 | 36.32 | 8.26 | 104 | 299 | 14.94 |
| story | 300 | 42.22 | 7.11 | 180 | 299 | 14.12 |

**S=512, max_new=1000** — mean 9.97 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 119.86 | 8.34 | 789 | 999 | 13.93 |
| fibonacci | 1000 | 86.17 | 11.60 | 661 | 999 | 13.75 |

**S=1024, max_new=100** — mean 28.69 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 6.89 | 14.52 | 56 | 99 | 17.51 |
| fibonacci | 100 | 1.68 | 59.63 | 14 | 99 | 17.21 |
| math | 100 | 3.78 | 26.44 | 28 | 99 | 18.76 |
| story | 100 | 7.05 | 14.19 | 52 | 99 | 18.58 |

**S=1024, max_new=300** — mean 16.06 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 21.09 | 14.22 | 142 | 301 | 18.68 |
| fibonacci | 300 | 12.94 | 23.19 | 81 | 302 | 19.62 |
| math | 300 | 17.32 | 17.32 | 91 | 300 | 18.59 |
| story | 300 | 31.53 | 9.52 | 162 | 300 | 18.30 |

**S=1024, max_new=1000** — mean 12.00 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 85.77 | 11.66 | 705 | 999 | 15.96 |
| fibonacci | 1000 | 81.05 | 12.34 | 533 | 999 | 18.00 |

**S=2048, max_new=100** — mean 24.05 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 9.56 | 10.46 | 56 | 99 | 27.16 |
| fibonacci | 100 | 2.34 | 42.74 | 14 | 99 | 27.18 |
| math | 100 | 3.37 | 29.64 | 28 | 99 | 24.16 |
| story | 100 | 7.48 | 13.37 | 52 | 99 | 25.07 |

**S=2048, max_new=300** — mean 13.32 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 30.37 | 9.88 | 142 | 301 | 27.73 |
| fibonacci | 300 | 17.49 | 17.15 | 81 | 302 | 27.67 |
| math | 300 | 19.43 | 15.44 | 91 | 300 | 27.51 |
| story | 300 | 27.72 | 10.82 | 162 | 300 | 25.92 |

**S=2048, max_new=1000** — mean 21.90 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 63.40 | 15.77 | 322 | 1006 | 25.75 |
| fibonacci | 1000 | 35.69 | 28.02 | 290 | 1001 | 23.78 |

**S=4096, max_new=100** — mean 21.47 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 7.60 | 13.16 | 56 | 99 | 38.86 |
| fibonacci | 100 | 2.32 | 43.10 | 14 | 99 | 40.39 |
| math | 100 | 5.22 | 19.16 | 28 | 99 | 40.17 |
| story | 100 | 9.57 | 10.45 | 52 | 99 | 41.67 |

**S=4096, max_new=300** — mean 16.39 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 28.38 | 10.57 | 142 | 301 | 44.78 |
| fibonacci | 300 | 14.20 | 21.12 | 81 | 302 | 44.78 |
| math | 300 | 14.39 | 20.84 | 91 | 300 | 43.32 |
| story | 300 | 23.02 | 13.03 | 162 | 300 | 40.08 |

**S=4096, max_new=1000** — mean 18.29 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 56.54 | 17.69 | 322 | 1006 | 41.91 |
| fibonacci | 1000 | 52.95 | 18.89 | 290 | 1001 | 43.17 |

</details>

## LUT6 per_tensor

| state_length \ max_new | 100 | 300 | 1000 |
|---:|---:|---:|---:|
| S=512 | 29.86 | 13.60 | 11.60 |
| S=1024 | 24.22 | 17.90 | 10.66 |
| S=2048 | 19.65 | 20.04 | 27.50 |
| S=4096 | 27.94 | 20.99 | 24.44 |

<details><summary>Per-prompt detail</summary>

**S=512, max_new=100** — mean 29.86 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 5.68 | 17.62 | 56 | 99 | 8.44 |
| fibonacci | 100 | 1.42 | 70.36 | 14 | 99 | 8.48 |
| math | 100 | 4.77 | 20.96 | 30 | 99 | 8.94 |
| story | 100 | 9.54 | 10.49 | 53 | 99 | 9.49 |

**S=512, max_new=300** — mean 13.60 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 31.04 | 9.67 | 156 | 299 | 9.03 |
| fibonacci | 300 | 20.69 | 14.50 | 103 | 299 | 8.49 |
| math | 300 | 18.96 | 15.82 | 108 | 299 | 8.43 |
| story | 300 | 20.83 | 14.40 | 175 | 299 | 8.33 |

**S=512, max_new=1000** — mean 11.60 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 90.74 | 11.02 | 796 | 999 | 8.26 |
| fibonacci | 1000 | 82.16 | 12.17 | 671 | 999 | 8.22 |

**S=1024, max_new=100** — mean 24.22 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 9.08 | 11.01 | 56 | 99 | 17.58 |
| fibonacci | 100 | 2.18 | 45.89 | 14 | 99 | 13.90 |
| math | 100 | 4.58 | 21.85 | 30 | 99 | 12.15 |
| story | 100 | 5.52 | 18.12 | 53 | 99 | 11.90 |

**S=1024, max_new=300** — mean 17.90 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 15.62 | 19.21 | 142 | 301 | 11.89 |
| fibonacci | 300 | 13.19 | 22.74 | 81 | 302 | 12.30 |
| math | 300 | 15.82 | 18.96 | 92 | 300 | 14.22 |
| story | 300 | 28.08 | 10.68 | 163 | 300 | 14.38 |

**S=1024, max_new=1000** — mean 10.66 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 106.45 | 9.39 | 710 | 999 | 12.86 |
| fibonacci | 1000 | 83.82 | 11.93 | 539 | 999 | 14.22 |

**S=2048, max_new=100** — mean 19.65 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 9.82 | 10.18 | 56 | 99 | 23.58 |
| fibonacci | 100 | 2.58 | 38.75 | 14 | 99 | 25.58 |
| math | 100 | 5.43 | 18.40 | 30 | 99 | 26.13 |
| story | 100 | 8.87 | 11.28 | 53 | 99 | 23.46 |

**S=2048, max_new=300** — mean 20.04 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 25.50 | 11.77 | 142 | 301 | 23.94 |
| fibonacci | 300 | 12.44 | 24.12 | 81 | 302 | 22.73 |
| math | 300 | 10.62 | 28.25 | 92 | 300 | 19.43 |
| story | 300 | 18.74 | 16.01 | 163 | 300 | 19.46 |

**S=2048, max_new=1000** — mean 27.50 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 37.99 | 26.32 | 323 | 999 | 19.44 |
| fibonacci | 1000 | 34.86 | 28.69 | 294 | 999 | 19.44 |

**S=4096, max_new=100** — mean 27.94 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 7.09 | 14.11 | 56 | 99 | 34.34 |
| fibonacci | 100 | 1.77 | 56.40 | 14 | 99 | 34.29 |
| math | 100 | 3.80 | 26.33 | 30 | 99 | 34.01 |
| story | 100 | 6.70 | 14.93 | 53 | 99 | 34.03 |

**S=4096, max_new=300** — mean 20.99 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 18.29 | 16.40 | 142 | 301 | 34.17 |
| fibonacci | 300 | 10.60 | 28.30 | 81 | 302 | 34.16 |
| math | 300 | 11.97 | 25.06 | 92 | 300 | 34.30 |
| story | 300 | 21.16 | 14.18 | 163 | 300 | 34.29 |

**S=4096, max_new=1000** — mean 24.44 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 42.76 | 23.38 | 323 | 999 | 34.17 |
| fibonacci | 1000 | 39.21 | 25.50 | 294 | 999 | 34.11 |

</details>

## LUT4 per_grouped_channel (group=8)

| state_length \ max_new | 100 | 300 | 1000 |
|---:|---:|---:|---:|
| S=512 | 33.89 | 20.95 | 12.35 |
| S=1024 | 32.74 | 24.14 | 14.20 |
| S=2048 | 30.51 | 22.54 | 26.17 |
| S=4096 | 26.90 | 20.00 | 23.25 |

<details><summary>Per-prompt detail</summary>

**S=512, max_new=100** — mean 33.89 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 5.83 | 17.14 | 58 | 99 | 8.36 |
| fibonacci | 100 | 1.51 | 66.23 | 15 | 99 | 8.35 |
| math | 100 | 2.93 | 34.17 | 29 | 105 | 8.34 |
| story | 100 | 5.55 | 18.03 | 55 | 99 | 8.41 |

**S=512, max_new=300** — mean 20.95 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 17.37 | 17.27 | 167 | 299 | 8.32 |
| fibonacci | 300 | 12.32 | 24.34 | 116 | 299 | 8.04 |
| math | 300 | 11.17 | 26.85 | 106 | 299 | 8.08 |
| story | 300 | 19.56 | 15.34 | 187 | 299 | 8.08 |

**S=512, max_new=1000** — mean 12.35 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 89.38 | 11.19 | 832 | 999 | 8.12 |
| fibonacci | 1000 | 73.96 | 13.52 | 686 | 999 | 8.07 |

**S=1024, max_new=100** — mean 32.74 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 6.05 | 16.54 | 58 | 99 | 11.95 |
| fibonacci | 100 | 1.56 | 64.00 | 15 | 99 | 12.00 |
| math | 100 | 3.03 | 33.00 | 29 | 105 | 12.04 |
| story | 100 | 5.74 | 17.41 | 55 | 99 | 12.00 |

**S=1024, max_new=300** — mean 24.14 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 15.77 | 19.03 | 148 | 299 | 12.00 |
| fibonacci | 300 | 9.86 | 30.43 | 91 | 299 | 12.01 |
| math | 300 | 9.81 | 30.59 | 91 | 300 | 11.95 |
| story | 300 | 18.16 | 16.52 | 169 | 300 | 11.96 |

**S=1024, max_new=1000** — mean 14.20 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 80.85 | 12.37 | 732 | 999 | 11.58 |
| fibonacci | 1000 | 62.39 | 16.03 | 562 | 999 | 11.40 |

**S=2048, max_new=100** — mean 30.51 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 6.48 | 15.42 | 58 | 99 | 19.50 |
| fibonacci | 100 | 1.68 | 59.61 | 15 | 99 | 19.42 |
| math | 100 | 3.25 | 30.74 | 29 | 105 | 19.59 |
| story | 100 | 6.15 | 16.25 | 55 | 99 | 19.45 |

**S=2048, max_new=300** — mean 22.54 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 16.89 | 17.77 | 148 | 299 | 19.51 |
| fibonacci | 300 | 10.56 | 28.42 | 91 | 299 | 19.56 |
| math | 300 | 10.51 | 28.53 | 91 | 300 | 19.54 |
| story | 300 | 19.45 | 15.42 | 169 | 300 | 19.57 |

**S=2048, max_new=1000** — mean 26.17 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 41.18 | 24.28 | 350 | 1006 | 19.55 |
| fibonacci | 1000 | 35.65 | 28.05 | 301 | 1006 | 19.47 |

**S=4096, max_new=100** — mean 26.90 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 7.35 | 13.61 | 58 | 99 | 34.25 |
| fibonacci | 100 | 1.90 | 52.52 | 15 | 99 | 34.54 |
| math | 100 | 3.69 | 27.10 | 29 | 105 | 34.66 |
| story | 100 | 6.96 | 14.36 | 55 | 99 | 34.33 |

**S=4096, max_new=300** — mean 20.00 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 19.07 | 15.73 | 148 | 299 | 34.25 |
| fibonacci | 300 | 11.88 | 25.24 | 91 | 299 | 34.26 |
| math | 300 | 11.84 | 25.33 | 91 | 300 | 34.25 |
| story | 300 | 21.94 | 13.68 | 169 | 300 | 34.28 |

**S=4096, max_new=1000** — mean 23.25 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 46.36 | 21.57 | 350 | 1006 | 34.27 |
| fibonacci | 1000 | 40.13 | 24.92 | 301 | 1006 | 34.40 |

</details>
