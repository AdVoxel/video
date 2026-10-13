#!/usr/bin/env python3
"""Packed alpha via clean-plate difference matte (numpy only, static camera).

pass 1: build a clean backdrop plate = per-pixel median of frames where the
        pixel looks like backdrop (close to the key color)
pass 2: alpha = smoothstep of distance to the plate; unmix the plate out of
        soft pixels; clamp blue spill; fill small enclosed holes; write
        color | mask side by side to x264.
"""
import argparse
import subprocess
import sys

import numpy as np


def probe(src):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate", "-of", "csv=p=0", src],
                       capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(o[0]), int(o[1]), o[2]


def read_frames(src, w, h, vf=None):
    cmd = ["ffmpeg", "-v", "error", "-i", src]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.stdout.close(); p.wait()


def box(a, r):
    for ax in (0, 1):
        c = np.cumsum(np.pad(a, [(r + 1, r) if i == ax else (0, 0) for i in range(a.ndim)],
                             mode="edge"), axis=ax)
        a = (np.take(c, np.arange(2 * r + 1, c.shape[ax]), axis=ax)
             - np.take(c, np.arange(0, c.shape[ax] - 2 * r - 1), axis=ax)) / (2 * r + 1)
    return a


def grow(m, k=1):
    o = m.copy()
    o[k:] |= m[:-k]; o[:-k] |= m[k:]; o[:, k:] |= m[:, :-k]; o[:, :-k] |= m[:, k:]
    return o


def erode(a, n):
    for _ in range(n):
        p = np.pad(a, 1, mode="edge")
        a = np.minimum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1],
                               p[1:-1, :-2], p[1:-1, 2:]])
    return a


def build_plate(src, w, h, key, ds, near):
    sw, sh = w // ds, h // ds
    stack = []
    for f in read_frames(src, sw, sh, f"scale={sw}:{sh}:flags=area"):
        stack.append(f)
    s = np.stack(stack).astype(np.float32)                       # N,h,w,3
    d = np.sqrt(((s - key) ** 2).sum(-1))
    s[d > near] = np.nan
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        plate = np.nanmedian(s, axis=0)                          # h,w,3
    seen = ~np.isnan(plate[..., 0])
    # pixels never seen as backdrop (always behind the stick/honey): fill from neighbours
    filled = np.where(seen[..., None], plate, 0)
    wgt = seen.astype(np.float32)
    r = 4
    while not wgt.all() and r < max(sw, sh):
        num = box(np.where(seen[..., None], plate, 0), r)
        den = box(seen.astype(np.float32), r)[..., None]
        est = np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)
        hole = ~seen & ~np.isnan(est[..., 0]) & (wgt == 0)
        filled[hole] = est[hole]; wgt[hole] = 1
        r *= 2
    filled[wgt == 0] = key
    filled = box(filled, 2)                                      # denoise the plate
    up = np.repeat(np.repeat(filled, ds, 0), ds, 1)
    up = np.pad(up, ((0, h - up.shape[0]), (0, w - up.shape[1]), (0, 0)), mode="edge")
    return box(up, ds).astype(np.float32), seen.mean()


def small_holes(alpha, thr, ds, max_area):
    """Enclosed transparent regions not reachable from the border and smaller than max_area px."""
    s = alpha[::ds, ::ds] < thr
    reach = np.zeros_like(s)
    reach[0], reach[-1], reach[:, 0], reach[:, -1] = s[0], s[-1], s[:, 0], s[:, -1]
    while True:
        nxt = grow(reach) & s
        if (nxt == reach).all():
            break
        reach = nxt
    enc = s & ~reach
    if not enc.any():
        return None
    # label enclosed blobs by repeated flood fill (they are few)
    keep = np.zeros_like(enc)
    todo = enc.copy()
    lim = max_area / (ds * ds)
    while todo.any():
        y, x = np.argwhere(todo)[0]
        blob = np.zeros_like(todo); blob[y, x] = True
        while True:
            nxt = grow(blob) & todo
            if (nxt == blob).all():
                break
            blob = nxt
        todo &= ~blob
        if blob.sum() <= lim:
            keep |= blob
    if not keep.any():
        return None
    keep = grow(grow(keep))
    return np.repeat(np.repeat(keep, ds, 0), ds, 1)[:alpha.shape[0], :alpha.shape[1]]


def inpaint(rgb, hm, al, ds, soft):
    small = rgb[::ds, ::ds]
    w = (~hm[::ds, ::ds]) * (al[::ds, ::ds] > 0.98).astype(np.float32)
    fillc = None
    for r in (8, 24, 64, 160):
        num = box(small * w[..., None], r); den = box(w, r)[..., None]
        est = num / np.maximum(den, 1e-4)
        fillc = est if fillc is None else np.where(fd > 1e-3, fillc, est)
        fd = den if r == 8 else np.maximum(fd, den)
        if (fd[hm[::ds, ::ds]] > 1e-3).all():
            break
    fillc = np.repeat(np.repeat(fillc, ds, 0), ds, 1)[:rgb.shape[0], :rgb.shape[1]]
    lum = lambda x: x @ np.array([0.299, 0.587, 0.114], np.float32)
    ratio = np.clip(lum(rgb) / np.maximum(lum(fillc), 1), 0.75, 1.35)[..., None]
    soft = soft[..., None]
    return rgb * (1 - soft) + np.clip(fillc * ratio, 0, 255) * soft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("out")
    p.add_argument("--key", default="3B98D2", help="backdrop color RRGGBB (plate seed)")
    p.add_argument("--near", type=float, default=70, help="plate: max RGB dist to count as backdrop")
    p.add_argument("--lo", type=float, default=12, help="dist to plate where alpha starts > 0")
    p.add_argument("--hi", type=float, default=90, help="dist to plate where alpha reaches 1")
    p.add_argument("--hole-area", type=int, default=20000, help="fill enclosed holes up to N px")
    p.add_argument("--erode", type=int, default=1)
    p.add_argument("--band", type=int, default=150, help="side bands with per-column plate correction (0 = off)")
    p.add_argument("--fill", default="A86436")
    p.add_argument("--crf", type=int, default=26)
    p.add_argument("--frames", type=int, default=0)
    a = p.parse_args()

    w, h, fps = probe(a.src)
    key = np.array([int(a.key[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    fill = np.array([int(a.fill[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    plate, cov = build_plate(a.src, w, h, key, 4, a.near)
    print(f"plate built ({cov * 100:.1f}% of pixels seen as backdrop)", file=sys.stderr)

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{2 * w}x{h}", "-r", fps, "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", a.out],
                           stdin=subprocess.PIPE)
    n = filled = 0
    for f in read_frames(a.src, w, h):
        rgb = f.astype(np.float32)
        # per-frame global drift of the backdrop (grade/exposure ramp): robust offset
        pf = plate
        sub = (slice(None, None, 8), slice(None, None, 8))
        for gate in (90, 25):
            dd = rgb[sub] - pf[sub]
            m = np.sqrt((dd ** 2).sum(-1)) < gate
            if m.sum() > 500:
                pf = pf + np.median(dd[m], axis=0)
        # per-column correction in the side bands (vignette that comes and goes with
        # dissolves); the subject never reaches these bands, tapered to 0 inwards
        band = a.band
        if band:
            ds = 4
            sm, pm = rgb[::ds, ::ds], pf[::ds, ::ds]
            res = sm - pm
            ok = (np.sqrt(((sm - key) ** 2).sum(-1)) < 70).astype(np.float32)
            bx = band // ds
            inb = np.zeros(ok.shape, np.float32); inb[:, :bx] = 1; inb[:, -bx:] = 1
            wgt = ok * inb
            num = box(res * wgt[..., None], 6); den = box(wgt, 6)[..., None]
            corr = np.where(den > 0.05, num / np.maximum(den, 1e-3), 0)
            # taper: full inside the bands, fading to 0 over another band width inward
            xs = np.arange(ok.shape[1], dtype=np.float32)
            dist = np.minimum(xs, ok.shape[1] - 1 - xs)
            ramp = np.clip(1 - (dist - bx) / bx, 0, 1)
            # inside the taper zone reuse the value at the band edge (same row)
            edge_l = corr[:, bx - 1:bx]; edge_r = corr[:, -bx:-bx + 1]
            inner = np.where((xs < ok.shape[1] / 2)[None, :, None], edge_l, edge_r)
            corr = np.where((dist < bx)[None, :, None], corr, inner) * ramp[None, :, None]
            corr = np.repeat(np.repeat(corr, ds, 0), ds, 1)[:h, :w]
            pf = pf + box(corr.astype(np.float32), ds)
        d = np.sqrt(((rgb - pf) ** 2).sum(-1))
        t = np.clip((d - a.lo) / (a.hi - a.lo), 0, 1)
        al = t * t * (3 - 2 * t)                                 # smoothstep
        # unmix: observed = al*fg + (1-al)*plate
        af = np.maximum(al, 0.04)[..., None]
        fg = np.clip((rgb - (1 - af) * pf) / af, 0, 255)
        fg[..., 2] = np.minimum(fg[..., 2], fg[..., :2].max(axis=2))
        hm = small_holes(al, 0.5, 4, a.hole_area)
        if hm is not None:
            # smooth full-res weight: removes the 4px stair-steps of the coarse mask
            soft = np.clip(box(box(hm.astype(np.float32), 4), 4) * 1.6 - 0.3, 0, 1)
            fg = inpaint(fg, hm, al, 4, soft)
            filled += int(hm.sum())
            al = np.maximum(al, soft)
        mask = erode((al * 255 + 0.5).astype(np.uint8), a.erode)
        af = mask.astype(np.float32)[..., None] / 255.0
        color = fg * af + fill * (1 - af)
        out = np.empty((h, 2 * w, 3), np.uint8)
        out[:, :w] = np.clip(color + 0.5, 0, 255).astype(np.uint8)
        out[:, w:] = mask[..., None]
        enc.stdin.write(out.tobytes())
        n += 1
        if a.frames and n >= a.frames:
            break
    enc.stdin.close(); enc.wait()
    print(f"{n} frames, avg hole px filled/frame {filled / max(n, 1):.0f}", file=sys.stderr)
    sys.exit(enc.returncode)


if __name__ == "__main__":
    main()
