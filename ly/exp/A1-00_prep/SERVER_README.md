# A1 在租用 GPU 服务器上的运行步骤

对应 `ly/commit.md` 中的 A1-00（预注册）、A1-01（代码准备）和 A1-02（运行）。

## 0. 机器要求

- **当前方案：2 台 H20 96GB，每台 1 卡。**
- H20 的 BF16 张量算力约为 4090 的 0.9 倍，FP32 约为 0.5 倍，显存带宽约为 4 倍。本实验的模型很小，速度与 4090 大致相当。
- 训练脚本全程 BF16 autocast，并开启 cuDNN autotune 和 TF32。

**每台机器：**

- CPU ≥ 16 核，内存 ≥ 64 GB，数据盘 ≥ 200 GB
- Ubuntu 22.04，CUDA ≥ 12.1，PyTorch ≥ 2.1（Hopper 架构 sm_90）
- 能访问 nlpr.ia.ac.cn（下载 HWDB）

**分工：**

- 两台各下载、预处理一份数据。如果平台有共享存储，只做一次，把 `ly/data/hwdb` 放在共享盘即可。
- 机器 A 跑种子 0，机器 B 跑种子 1。
- HWDB 跑完后，任意一台接着做 BCTR 续训试验（A1-02 之后的条目）。

## 1. 上传代码包

本机运行：

```bash
bash ly/scripts/make_a1_bundle.sh
```

得到 `ly/exp/A1-00_prep/a1_bundle.tar.gz`，只有几 MB。上传到服务器后，在工作目录（下文记作 `$W`）解压：

```bash
tar xzf a1_bundle.tar.gz
```

包内保留原相对路径：

- `ly/scripts/`
- `ly/exp/A1-00_prep/`
- `OpenOCR-main/` 下的 `IDS_V7.txt` 和 `train_character_frequency.json`

## 2. 环境

```bash
pip install rapidfuzz numpy pillow
sudo apt-get install -y unalz
```

HWDB 的内层压缩包是 ALZ 格式，需要 `unalz`。没有 sudo 时，可以用 `unar` 代替。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 下载 HWDB（CASIA 官方，仅限学术使用，论文须引用 Liu et al., ICDAR 2011）

已于 2026-10-12 在服务器上核实（链接见下载页 `Download.html`）：

- http 不通，只能用 https；证书域名不匹配，下载时要加 `--no-check-certificate`。
- 文件名不是网上流传的 `HWDB1.xtrn_gnt.zip`，而是下面这些。
- 2020 年以后，学术研究使用不需要申请表。

```bash
mkdir -p /root/autodl-tmp/raw && cd /root/autodl-tmp/raw
base=https://www.nlpr.ia.ac.cn/databases/Download
for f in Offline/CharData/Gnt1.0TrainPart1.zip Offline/CharData/Gnt1.0TrainPart2.zip Offline/CharData/Gnt1.0TrainPart3.zip \
         Offline/CharData/Gnt1.0Test.zip Offline/CharData/Gnt1.1TrainPart1.zip Offline/CharData/Gnt1.1TrainPart2.zip \
         Offline/CharData/Gnt1.1Test.zip competition/competition-gnt.zip; do
  wget -c --no-check-certificate "$base/$f"
done
```

官网标注的大小：

| 文件 | 大小 |
|---|---:|
| Gnt1.0TrainPart1 / Part2 / Part3 | 962 / 973 / 983 MB |
| Gnt1.1TrainPart1 / Part2 | 897 / 943 MB |
| competition-gnt | 448 MB |
| Gnt1.0Test、Gnt1.1Test | 未标注 |

下载后：

```bash
for f in *.zip; do unzip -t "$f" > /dev/null && echo "ok $f"; done
```

解压后看里面是 `.gnt` 还是 `.alz`。如果是 `.alz`，先 `apt-get install -y unalz` 再解。最后整理成下面的目录，每个目录里都是 `.gnt` 文件：

- `HWDB1.0trn/`（Train Part1–3）
- `HWDB1.1trn/`（Train Part1–2）
- `HWDB1.0tst/`、`HWDB1.1tst/`
- `competition_gnt/`

## 4. 预处理

在 `$W` 下运行。

```bash
python ly/scripts/a1_hwdb_prepare.py convert --name train --gnt-dirs raw/HWDB1.0trn raw/HWDB1.1trn --out ly/data/hwdb
python ly/scripts/a1_hwdb_prepare.py convert --name val   --gnt-dirs raw/HWDB1.0tst raw/HWDB1.1tst --out ly/data/hwdb
python ly/scripts/a1_hwdb_prepare.py convert --name test  --gnt-dirs raw/competition_gnt          --out ly/data/hwdb
python ly/scripts/a1_hwdb_prepare.py classes  --out ly/data/hwdb
python ly/scripts/a1_hwdb_prepare.py longtail --out ly/data/hwdb --imbalance 100 --nmax 500
```

- `classes` 必须输出 3755 类；如果和本地的 GB2312 一级字表不一致，脚本会直接报错。
- `longtail` 会打印 many / medium / few 的类别数。如果报"某类样本不足"，就用更小的 `--nmax` 重跑，并在 `commit.md` 里记录。

## 5. 运行 A1-02

先单跑一个 ce 试 1 个 epoch，确认速度和显存：

```bash
python ly/scripts/a1_train_hccr.py train --data ly/data/hwdb --split lt100 --loss ce --seed 0 --epochs 1 --out /tmp/speedtest
```

按估算，每个 epoch 约 0.5–1 分钟，每组 30 个 epoch 约 20–30 分钟，每台机器 8 组约 3–4 小时。

正式运行：

```bash
nohup bash ly/scripts/a1_run_queue.sh 0 > queue_s0.log 2>&1 &   # 机器 A
nohup bash ly/scripts/a1_run_queue.sh 1 > queue_s1.log 2>&1 &   # 机器 B
```

H20 显存有 96 GB，单个实验只用几 GB。如果 `nvidia-smi` 显示 GPU 利用率长期低于 60%，可以在同一台机器上再并行启动一个队列，例如两台都跑完后再加种子 2；也可以留给 BCTR 试验。

## 6. 带回本地

只需要以下文件，每个实验几 KB；不要带回 `model.pt` 和数据：

- `ly/exp/A1-02_hwdb_lt/*/result.json`
- `ly/exp/A1-02_hwdb_lt/*/log.json`
- `ly/exp/A1-02_hwdb_lt/*/args.json`
- `ly/data/hwdb/train_lt100_summary.json`
