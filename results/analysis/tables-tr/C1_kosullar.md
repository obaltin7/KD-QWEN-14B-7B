| Öğe | Değer |
|---|---|
| Öğretmen (T) | Qwen/Qwen2.5-14B-Instruct (dondurulmuş, bf16) |
| Öğrenci (S) | Qwen/Qwen2.5-7B-Instruct + LoRA |
| Eğitim verisi | GSM8K train_rest, n = 7173 |
| En uzun örnek / kırpılan | 446 belirteç / 0 |
| Test | GSM8K test, n = 1319 |
| İstem | Question: {soru}\nAnswer: — sohbet şablonu yok |
| LoRA | r = 16, alfa = 32, dropout = 0,05, hedef: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Eğitilebilir LoRA parametresi | 40.370.176 |
| Optimizasyon | AdamW (wd = 0), lr = 0.0001, %3 doğrusal ısınma + kosinüs, gradyan kırpma 1,0 |
| Yığın | etkin 16 (mikro 4) |
| Dönem / adım | 2 / 898 (ısınma 27) |
| Dizi uzunluğu | en fazla 512 |
| KD (S2, S3) | τ = 2, α = 0,5, ileri KL, ilk 151.665 sözlük indeksi |
| Gizli durum (S3) | β = 1, çiftler 7↔12, 14↔24, 21↔36, 28↔48, W: 3584→5120 (paylaşılan) |
| Tohumlar | 42, 1234 |
| Süre / en yüksek bellek — S1 (tohum 42) | 22,0 dk / 17,5 GB |
| Süre / en yüksek bellek — S2 (tohum 42) | 30,1 dk / 49,2 GB |
| Süre / en yüksek bellek — S3 (tohum 42) | 31,9 dk / 50,2 GB |
| Süre / en yüksek bellek — S1 (tohum 1234) | 21,7 dk / 17,4 GB |
| Süre / en yüksek bellek — S2 (tohum 1234) | 30,5 dk / 49,1 GB |
| Süre / en yüksek bellek — S3 (tohum 1234) | 31,5 dk / 50,1 GB |
