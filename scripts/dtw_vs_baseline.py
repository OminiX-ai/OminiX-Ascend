"""DTW log-mel similarity matrix for the native-TTS quality gate.

Input: a directory containing paired WAVs named `<uid>.<backend>.wav`
(e.g. `short.llama.wav`, `short.native.wav`). Outputs a TSV with per-utterance
DTW similarity between each `<uid>.llama.wav` and `<uid>.native.wav`, plus
duration stats and a pass/fail verdict against the 0.85 gate.

Usage: python3 dtw_vs_baseline.py /path/to/quality_dir
"""
import os, sys, glob, re
import wave, array, math

def load_mono(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        nchan = w.getnchannels()
        raw = w.readframes(n)
    s = array.array("h", raw)
    if nchan == 2:
        s = array.array("h", [(s[i] + s[i + 1]) // 2 for i in range(0, len(s), 2)])
    return [x / 32768.0 for x in s], sr


def fft(x):
    n = len(x)
    if n == 1:
        return [(x[0], 0.0)]
    even = fft([x[i] for i in range(0, n, 2)])
    odd = fft([x[i] for i in range(1, n, 2)])
    out = [(0.0, 0.0)] * n
    for k in range(n // 2):
        a = -2 * math.pi * k / n
        c, s = math.cos(a), math.sin(a)
        tre = c * odd[k][0] - s * odd[k][1]
        tim = c * odd[k][1] + s * odd[k][0]
        out[k] = (even[k][0] + tre, even[k][1] + tim)
        out[k + n // 2] = (even[k][0] - tre, even[k][1] - tim)
    return out


def mel_bands(power, sr, n_fft, n_mels=40):
    def hz_to_mel(h): return 2595 * math.log10(1 + h / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mel_pts = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hz_pts = [mel_to_hz(m) for m in mel_pts]
    bins = [int(h * n_fft / sr) for h in hz_pts]
    out = []
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        e = 0.0
        for k in range(lo, hi + 1):
            if k >= len(power): break
            if k < mid: w = (k - lo) / max(1, mid - lo)
            else: w = (hi - k) / max(1, hi - mid)
            if w > 0: e += w * power[k]
        out.append(e)
    return out


def extract_mel(signal, sr, n_fft=512, hop=256):
    frames = []
    for start in range(0, len(signal) - n_fft, hop):
        win = signal[start:start + n_fft]
        win = [win[i] * 0.5 * (1 - math.cos(2 * math.pi * i / (n_fft - 1))) for i in range(n_fft)]
        X = fft(win)
        power = [r * r + im * im for (r, im) in X[:n_fft // 2 + 1]]
        mel = mel_bands(power, sr, n_fft)
        log_mel = [math.log(e + 1e-10) for e in mel]
        frames.append(log_mel)
    return frames


def cos_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0: return 0
    return dot / (na * nb)


def dtw_similarity(seq_a, seq_b):
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0: return 0.0
    INF = float("inf")
    band = max(n, m) // 4 + 5
    cost = [[INF] * m for _ in range(n)]
    cost[0][0] = 1 - cos_sim(seq_a[0], seq_b[0])
    for i in range(n):
        jmin = max(0, i - band); jmax = min(m, i + band)
        for j in range(jmin, jmax):
            if i == 0 and j == 0: continue
            prev = INF
            if i > 0: prev = min(prev, cost[i - 1][j])
            if j > 0: prev = min(prev, cost[i][j - 1])
            if i > 0 and j > 0: prev = min(prev, cost[i - 1][j - 1])
            cost[i][j] = prev + (1 - cos_sim(seq_a[i], seq_b[j]))
    i, j = n - 1, m - 1
    sim_sum = 0
    plen = 0
    while i > 0 or j > 0:
        sim_sum += cos_sim(seq_a[i], seq_b[j]); plen += 1
        opts = []
        if i > 0 and j > 0: opts.append((cost[i - 1][j - 1], i - 1, j - 1))
        if i > 0: opts.append((cost[i - 1][j], i - 1, j))
        if j > 0: opts.append((cost[i][j - 1], i, j - 1))
        _, i, j = min(opts)
    sim_sum += cos_sim(seq_a[0], seq_b[0]); plen += 1
    return sim_sum / plen


def main():
    if len(sys.argv) < 2:
        print("usage: dtw_vs_baseline.py QUALITY_DIR", file=sys.stderr); sys.exit(1)
    d = sys.argv[1]
    pairs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.wav"))):
        m = re.match(r"(?P<uid>[^.]+)\.(?P<bk>llama|native)\.wav$", os.path.basename(f))
        if not m: continue
        pairs.setdefault(m["uid"], {})[m["bk"]] = f

    print("uid\tllama_dur_s\tnative_dur_s\tDTW\tgate_pass")
    gate = 0.85
    n_pass = 0; n_total = 0
    for uid in sorted(pairs):
        p = pairs[uid]
        if "llama" not in p or "native" not in p:
            print(f"{uid}\t-\t-\t-\tMISSING"); continue
        la, sr_a = load_mono(p["llama"])
        na, sr_b = load_mono(p["native"])
        mel_a = extract_mel(la, sr_a)
        mel_b = extract_mel(na, sr_b)
        sim = dtw_similarity(mel_a, mel_b)
        pass_ = sim >= gate
        n_total += 1
        if pass_: n_pass += 1
        print(f"{uid}\t{len(la)/sr_a:.2f}\t{len(na)/sr_b:.2f}\t{sim:.4f}\t{'PASS' if pass_ else 'FAIL'}")
    print(f"\nAggregate: {n_pass}/{n_total} utterances pass the {gate} DTW gate")
    print("Note: DTW alone is necessary but NOT sufficient — the contract also")
    print("requires a user-ear pass on every utterance. Listen and verify.")


if __name__ == "__main__":
    main()
