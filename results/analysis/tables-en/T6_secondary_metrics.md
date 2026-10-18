| Metric | Hypothesis | Comparison | Seed | Acc. A % | Acc. B % | Difference (points) | Bootstrap 95% CI | b (only A) | c (only B) | p (raw) | p (Holm) | Significant (α = 0.05) | Direction | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-shot strict | H1 | S2−S1 | 42 | 34.3 | 76.9 | −42.5 | [−45.8, −39.3] | 74 | 635 | 4.3·10^-112 | 1.7·10^-111 | yes | unfavourable | not supported (significant in the opposite direction) (descriptive) |
| 0-shot strict | H2 | S3−S2 | 42 | 33.7 | 34.3 | −0.7 | [−3.2, 1.9] | 140 | 149 | 0.6380 | 0.6380 | no | unfavourable | not supported (descriptive) |
| 0-shot strict | H1 | S2−S1 | 1234 | 35.5 | 75.8 | −40.3 | [−43.6, −37.1] | 83 | 615 | 3.0·10^-101 | 9.1·10^-101 | yes | unfavourable | not supported (significant in the opposite direction) (descriptive) |
| 0-shot strict | H2 | S3−S2 | 1234 | 32.8 | 35.5 | −2.7 | [−5.1, −0.1] | 127 | 162 | 0.0453 | 0.0906 | no | unfavourable | not supported (descriptive) |
| 4-shot flexible | H1 | S2−S1 | 42 | 80.8 | 76.5 | 4.3 | [2.0, 6.6] | 149 | 92 | 0.0003 | 0.0009 | yes | favourable | supported (descriptive) |
| 4-shot flexible | H2 | S3−S2 | 42 | 78.5 | 80.8 | −2.4 | [−4.2, −0.5] | 60 | 91 | 0.0144 | 0.0144 | yes | unfavourable | not supported (significant in the opposite direction) (descriptive) |
| 4-shot flexible | H1 | S2−S1 | 1234 | 82.6 | 76.6 | 5.9 | [3.6, 8.2] | 162 | 84 | 7.5·10^-7 | 3.0·10^-6 | yes | favourable | supported (descriptive) |
| 4-shot flexible | H2 | S3−S2 | 1234 | 79.8 | 82.6 | −2.8 | [−4.7, −0.9] | 67 | 104 | 0.0057 | 0.0115 | yes | unfavourable | not supported (significant in the opposite direction) (descriptive) |
| 4-shot strict | H1 | S2−S1 | 42 | 80.4 | 76.5 | 3.9 | [1.7, 6.2] | 146 | 94 | 0.0010 | 0.0029 | yes | favourable | supported (descriptive) |
| 4-shot strict | H2 | S3−S2 | 42 | 78.1 | 80.4 | −2.4 | [−4.2, −0.5] | 62 | 93 | 0.0157 | 0.0295 | yes | unfavourable | not supported (significant in the opposite direction) (descriptive) |
| 4-shot strict | H1 | S2−S1 | 1234 | 81.6 | 75.7 | 5.8 | [3.4, 8.2] | 171 | 94 | 2.6·10^-6 | 1.0·10^-5 | yes | favourable | supported (descriptive) |
| 4-shot strict | H2 | S3−S2 | 1234 | 79.0 | 81.6 | −2.6 | [−4.5, −0.5] | 75 | 109 | 0.0148 | 0.0295 | yes | unfavourable | not supported (significant in the opposite direction) (descriptive) |

Secondary metrics (prereg §5). For each metric 4 tests in the H1/H2 structure, Holm within the metric; decisions are descriptive (the pre-registered decision uses the primary metric).
