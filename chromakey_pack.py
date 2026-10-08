#!/usr/bin/env python3
"""
chromakey_pack.py — remove fundo de cor de um video e entrega-o em varios formatos.

Offline por completo: so precisa de ffmpeg e ffprobe no PATH.
Sem dependencias de Python para alem da biblioteca padrao.

Modos de saida:
  packed   MP4 H.264 opaco com cor + mascara lado a lado (recompor no cliente)
  webm     WebM VP9 com alpha nativo (Chrome, Firefox, Edge)
  prores   MOV ProRes 4444 com alpha (edicao)
  baked    MP4 H.264 com o fundo substituido por uma cor solida

Exemplos:
  python3 chromakey_pack.py entrada.mov -o saida.mp4
  python3 chromakey_pack.py entrada.mov -o saida.mp4 --max-mb 3
  python3 chromakey_pack.py entrada.mov -o saida.webm --mode webm
  python3 chromakey_pack.py entrada.mov -o saida.mp4 --mode baked --bg-color 0xFFFFFF
  python3 chromakey_pack.py entrada.mov -o saida.mp4 --color 0x128749 --key colorkey --similarity 0.34
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass

# ---------------------------------------------------------------- utilitarios


def morrer(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"erro: {msg}", file=sys.stderr)
    raise SystemExit(1)


def exigir_ferramentas() -> None:
    em_falta = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if em_falta:
        morrer(f"nao encontrei no PATH: {', '.join(em_falta)}")


def correr(cmd: list[str], captura: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=captura, check=False)


def hex_para_rgb(s: str) -> tuple[int, int, int]:
    t = s.strip().lower().removeprefix("0x").removeprefix("#")
    if len(t) != 6:
        morrer(f"cor invalida: {s!r} (esperado 0xRRGGBB)")
    try:
        v = int(t, 16)
    except ValueError:
        morrer(f"cor invalida: {s!r}")
    return (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF


def rgb_para_hex(c: tuple[int, int, int]) -> str:
    return f"0x{c[0]:02X}{c[1]:02X}{c[2]:02X}"


def escapar(p: str) -> str:
    """Escapa um caminho para uso dentro de um filtro ffmpeg."""
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


# ------------------------------------------------------------------- sondagem


@dataclass
class Video:
    caminho: str
    largura: int
    altura: int
    fps: float
    duracao: float
    n_frames: int
    pix_fmt: str
    codec: str


def sondar(caminho: str) -> Video:
    if not os.path.isfile(caminho):
        morrer(f"ficheiro nao encontrado: {caminho}")
    r = correr([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name:format=duration",
        "-of", "json", caminho,
    ])
    if r.returncode != 0:
        morrer(f"ffprobe falhou: {r.stderr.decode(errors='replace').strip()}")
    d = json.loads(r.stdout or b"{}")
    if not d.get("streams"):
        morrer("nenhuma stream de video encontrada")
    s = d["streams"][0]
    num, _, den = (s.get("r_frame_rate") or "24/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 24.0
    dur = float(d.get("format", {}).get("duration") or 0.0)
    try:
        n = int(s.get("nb_frames") or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        n = max(1, int(round(dur * fps)))
    return Video(caminho, int(s["width"]), int(s["height"]), fps, dur, n,
                 s.get("pix_fmt", "?"), s.get("codec_name", "?"))


def frame_cru(caminho: str, indice: int, larg: int, alt: int,
              filtros: str = "", fmt: str = "rgb24") -> bytes:
    """Devolve um frame de um video como bytes crus, ja redimensionado."""
    canais = 4 if fmt == "rgba" else 3
    cadeia = f"select=\'eq(n\\,{indice})\'"
    if filtros:
        cadeia += "," + filtros
    cadeia += f",scale={larg}:{alt},format={fmt}"
    r = correr([
        "ffmpeg", "-v", "error", "-i", caminho, "-vf", cadeia,
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", fmt, "-",
    ])
    esperado = larg * alt * canais
    if r.returncode != 0 or len(r.stdout) < esperado:
        return b""
    return r.stdout[:esperado]


def imagem_crua(caminho: str, larg: int, alt: int,
                filtros: str = "", fmt: str = "rgb24") -> bytes:
    """O mesmo, mas a partir de uma imagem ja extraida (muito mais rapido)."""
    canais = 4 if fmt == "rgba" else 3
    cadeia = (filtros + "," if filtros else "") + f"scale={larg}:{alt},format={fmt}"
    r = correr([
        "ffmpeg", "-v", "error", "-i", caminho, "-vf", cadeia,
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", fmt, "-",
    ])
    esperado = larg * alt * canais
    if r.returncode != 0 or len(r.stdout) < esperado:
        return b""
    return r.stdout[:esperado]


def extrair_amostras(v: Video, n: int, larg: int, pasta: str) -> list[str]:
    """Extrai n frames espalhados pelo video, uma unica passagem de descodificacao.

    Evita reabrir e reprocurar no original a cada avaliacao, o que em ficheiros
    grandes (HD, 10 bits, HEVC) domina completamente o tempo de execucao.
    """
    alt = max(2, int(larg * v.altura / v.largura))
    alt -= alt % 2
    passo = max(1, v.n_frames // (n + 1))
    expr = "+".join("eq(n\\," + str(passo * k) + ")" for k in range(1, n + 1))
    padrao = os.path.join(pasta, "am%03d.png")
    r = correr([
        "ffmpeg", "-v", "error", "-y", "-i", v.caminho,
        "-vf", f"select=\'{expr}\',scale={larg}:{alt}",
        "-vsync", "0", padrao,
    ])
    if r.returncode != 0:
        morrer("nao consegui extrair frames de amostra")
    saidas = sorted(
        os.path.join(pasta, f) for f in os.listdir(pasta) if f.startswith("am")
    )
    if not saidas:
        morrer("nenhum frame de amostra foi extraido")
    return saidas


# ------------------------------------------------------- deteccao da cor de fundo


def detectar_cor(amostras: list[str], larg: int, alt: int,
                 margem: float = 0.04) -> tuple[int, int, int]:
    """Cor dominante nas margens dos frames de amostra."""
    mx, my = max(1, int(larg * margem)), max(1, int(alt * margem))
    contagem: Counter = Counter()
    buffers = []

    for caminho in amostras:
        buf = imagem_crua(caminho, larg, alt)
        if not buf:
            continue
        buffers.append(buf)
        for y in range(alt):
            borda_v = y < my or y >= alt - my
            for x in range(larg):
                if not (borda_v or x < mx or x >= larg - mx):
                    continue
                o = (y * larg + x) * 3
                contagem[(buf[o] >> 3, buf[o + 1] >> 3, buf[o + 2] >> 3)] += 1

    if not contagem:
        morrer("nao consegui ler as amostras para detectar a cor do fundo")

    balde, _ = contagem.most_common(1)[0]
    soma, total = [0, 0, 0], 0
    for buf in buffers:
        for p in range(0, len(buf), 3):
            if (buf[p] >> 3, buf[p + 1] >> 3, buf[p + 2] >> 3) == balde:
                soma[0] += buf[p]; soma[1] += buf[p + 1]; soma[2] += buf[p + 2]
                total += 1
    if total == 0:
        return (balde[0] << 3, balde[1] << 3, balde[2] << 3)
    return tuple(x // total for x in soma)  # type: ignore[return-value]


def cor_do_sujeito(amostras: list[str], cor: tuple[int, int, int], tipo: str,
                   sim: float, blend: float, lg: int, at: int) -> tuple[int, int, int]:
    """Cor media das zonas opacas — usada para preencher o fundo do plano de cor.

    Deixar o fundo na cor original do ecra faz com que a subamostragem de croma
    do H.264 a espalhe para as margens do sujeito, criando um halo. Preencher
    com um tom proximo do sujeito torna esse sangramento invisivel.
    """
    filtro = f"{tipo}={rgb_para_hex(cor)}:{sim:.3f}:{blend:.3f}"
    soma, total = [0, 0, 0], 0
    for caminho in amostras:
        buf = imagem_crua(caminho, lg, at, filtros=filtro, fmt="rgba")
        if not buf:
            continue
        for p in range(0, len(buf), 4):
            if buf[p + 3] > 240:
                soma[0] += buf[p]; soma[1] += buf[p + 1]; soma[2] += buf[p + 2]
                total += 1
    if total == 0:
        return (128, 128, 128)
    return tuple(x // total for x in soma)  # type: ignore[return-value]


# ------------------------------------------------------------ avaliacao do key


def _mascaras(buf: bytes, cor: tuple[int, int, int]) -> tuple[list[int], list[int]]:
    """Indices de pixeis claramente do fundo e claramente do sujeito."""
    fundo, sujeito = [], []
    for p in range(0, len(buf), 3):
        d = abs(buf[p] - cor[0]) + abs(buf[p + 1] - cor[1]) + abs(buf[p + 2] - cor[2])
        if d < 60:
            fundo.append(p // 3)
        elif d > 200:
            sujeito.append(p // 3)
    return fundo, sujeito


def avaliar(amostras: list[str], cor: tuple[int, int, int], tipo: str, sim: float,
            blend: float, lg: int, at: int,
            despill: str = "green") -> tuple[float, float, float]:
    """Devolve (residuo_fundo, retencao_sujeito, halo) medios."""
    filtro = f"{tipo}={rgb_para_hex(cor)}:{sim:.3f}:{blend:.3f}"
    canal = 2 if despill == "blue" else 1   # indice RGB do canal contaminante
    outros = (0, 1) if despill == "blue" else (0, 2)
    res_t = ret_t = halo_t = 0.0
    contados = 0

    for caminho in amostras:
        orig = imagem_crua(caminho, lg, at)
        keyed = imagem_crua(caminho, lg, at, filtros=filtro, fmt="rgba")
        if not orig or not keyed:
            continue
        fundo, sujeito = _mascaras(orig, cor)
        if not fundo or not sujeito:
            continue

        res = sum(1 for k in fundo if keyed[k * 4 + 3] > 32) / len(fundo)
        ret = sum(1 for k in sujeito if keyed[k * 4 + 3] > 200) / len(sujeito)
        vis = [k for k in sujeito if keyed[k * 4 + 3] > 200]
        halo = 0.0
        if vis:
            halo = sum(
                1 for k in vis
                if keyed[k * 4 + canal] > keyed[k * 4 + outros[0]] + 10
                and keyed[k * 4 + canal] > keyed[k * 4 + outros[1]] + 10
            ) / len(vis)

        res_t += res
        ret_t += ret
        halo_t += halo
        contados += 1

    if contados == 0:
        return 1.0, 0.0, 1.0
    return res_t / contados, ret_t / contados, halo_t / contados


def afinar(amostras: list[str], cor: tuple[int, int, int], tipos: list[str],
           blend: float, verboso: bool, despill: str,
           lg: int, at: int) -> tuple[str, float]:
    """Varre tipo de filtro e similarity, escolhe o menor custo."""
    melhor = None
    for tipo in tipos:
        for sim in [round(0.04 + 0.03 * k, 3) for k in range(0, 15)]:
            res, ret, halo = avaliar(amostras, cor, tipo, sim, blend, lg, at, despill)
            custo = res * 3.0 + (1.0 - ret) + halo * 0.5
            if verboso:
                print(f"    {tipo:9s} sim={sim:.2f}  fundo={res*100:5.2f}%  "
                      f"sujeito={ret*100:6.2f}%  halo={halo*100:5.2f}%  custo={custo:.3f}")
            if melhor is None or custo < melhor[0]:
                melhor = (custo, tipo, sim, res, ret, halo)

    if melhor is None:
        morrer("a afinacao automatica falhou; indica --key e --similarity a mao")
    _, tipo, sim, res, ret, halo = melhor
    print(f"  escolhido: {tipo} similarity={sim:.2f} "
          f"(fundo residual {res*100:.2f}%, sujeito preservado {ret*100:.1f}%, halo {halo*100:.2f}%)")
    return tipo, sim


# ------------------------------------------------------------------- filtragem


def tipo_despill(cor: tuple[int, int, int]) -> str:
    """O filtro despill so conhece 'green' e 'blue'; decide pelo canal dominante."""
    return "blue" if cor[2] >= cor[1] else "green"


def cadeia_key(cor: tuple[int, int, int], tipo: str, sim: float, blend: float,
               mix: float, expand: float, fps: float | None, escala: str | None,
               despill: str) -> str:
    # as escalas por omissao do filtro despill assumem greenscreen (green=-1,
    # blue=0); para bluescreen tem de se trocar, senao o filtro nao faz nada
    escalas = "blue=-1:green=0" if despill == "blue" else "green=-1:blue=0"
    partes = [f"{tipo}={rgb_para_hex(cor)}:{sim:.3f}:{blend:.3f}",
              f"despill=type={despill}:mix={mix:.2f}:expand={expand:.2f}:{escalas}"]
    if fps:
        partes.insert(0, f"fps={fps:g}")
    if escala:
        partes.append(f"scale={escala}")
    return ",".join(partes)


# ------------------------------------------------------------------ codificacao


def codificar(v: Video, saida: str, modo: str, chave: str, crf: int,
              bg: tuple[int, int, int], layout: str, verboso: bool,
              erode: int = 0, fill: tuple[int, int, int] | None = None) -> None:
    base = ["ffmpeg", "-v", "error" if not verboso else "info", "-y", "-i", v.caminho]

    if modo == "packed":
        empilha = "hstack=inputs=2" if layout == "h" else "vstack=inputs=2"
        chan = f"[0:v]{chave},format=rgba"
        if erode > 0:
            # encolhe o alpha uns pixeis: as margens sao onde a cor do ecra
            # ainda contamina o sujeito, e o despill nao as limpa por completo
            er = ",".join(["erosion"] * erode)
            chan += f",split[m0][m1];[m0]alphaextract,{er}[me];[m1][me]alphamerge"
        chan += ",split[c0][a0];"
        if fill is not None:
            cor_fill = f"0x{fill[0]:02X}{fill[1]:02X}{fill[2]:02X}"
            pre = (f"color=c={cor_fill}:s={v.largura}x{v.altura}:r={v.fps:g}[bgfill];")
            meio = "[bgfill][c0]overlay=shortest=1,format=yuv420p[c];"
        else:
            pre = ""
            meio = "[c0]format=yuv420p[c];"
        fc = (pre + chan + meio +
              f"[a0]alphaextract,format=yuv420p[a];"
              f"[c][a]{empilha}[out]")
        cmd = base + ["-filter_complex", fc, "-map", "[out]",
                      "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", saida]

    elif modo == "webm":
        cmd = base + ["-vf", f"{chave},format=yuva420p",
                      "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                      "-auto-alt-ref", "0", "-b:v", "0", "-crf", str(crf),
                      "-row-mt", "1", "-an", saida]

    elif modo == "prores":
        cmd = base + ["-vf", f"{chave},format=yuva444p10le",
                      "-c:v", "prores_ks", "-profile:v", "4444",
                      "-pix_fmt", "yuva444p10le", "-alpha_bits", "8",
                      "-qscale:v", "13", "-vendor", "apl0", "-an", saida]

    elif modo == "baked":
        cor = f"0x{bg[0]:02X}{bg[1]:02X}{bg[2]:02X}"
        fc = (f"color=c={cor}:s={v.largura}x{v.altura}:r={v.fps:g}[bg];"
              f"[0:v]{chave}[fg];[bg][fg]overlay=shortest=1,format=yuv420p[out]")
        cmd = base + ["-filter_complex", fc, "-map", "[out]",
                      "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", saida]
    else:
        morrer(f"modo desconhecido: {modo}")

    r = correr(cmd, captura=not verboso)
    if r.returncode != 0:
        detalhe = (r.stderr or b"").decode(errors="replace").strip()[-800:]
        morrer(f"a codificacao falhou:\n{detalhe}")


def mb(caminho: str) -> float:
    return os.path.getsize(caminho) / (1024 * 1024)


# ------------------------------------------------------------------ verificacao


def verificar_packed(saida: str, layout: str) -> None:
    v = sondar(saida)
    if layout == "h":
        lg, at = v.largura // 2, v.altura
        desloc = lg
    else:
        lg, at = v.largura, v.altura // 2
        desloc = 0

    amostra = max(1, v.n_frames // 2)
    buf = frame_cru(saida, amostra, v.largura, v.altura)
    if not buf:
        print("  aviso: nao consegui reler o resultado para verificar")
        return

    def px(x: int, y: int) -> tuple[int, int, int]:
        o = (y * v.largura + x) * 3
        return buf[o], buf[o + 1], buf[o + 2]

    if layout == "h":
        mascara = [px(desloc + x, y) for x in range(0, lg, lg // 16) for y in range(0, at, at // 16)]
    else:
        mascara = [px(x, at + y) for x in range(0, lg, lg // 16) for y in range(0, at, at // 16)]

    claros = sum(1 for c in mascara if c[0] > 200)
    escuros = sum(1 for c in mascara if c[0] < 55)
    print(f"  mascara: {claros} amostras opacas, {escuros} transparentes "
          f"de {len(mascara)}")
    if claros == 0:
        print("  AVISO: a mascara nao tem zonas opacas — o similarity pode estar alto demais")
    if escuros == 0:
        print("  AVISO: a mascara nao tem zonas transparentes — o fundo pode nao ter sido removido")


# ------------------------------------------------------------------------ main


def main() -> None:
    p = argparse.ArgumentParser(
        description="Remove fundo de cor e entrega packed / webm / prores / baked.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("entrada")
    p.add_argument("-o", "--output", required=True, help="ficheiro de saida")
    p.add_argument("--mode", default="packed",
                   choices=["packed", "webm", "prores", "baked"])

    p.add_argument("--color", default="auto",
                   help="cor do fundo, 0xRRGGBB, ou 'auto' (omissao)")
    p.add_argument("--key", default="auto", choices=["auto", "chromakey", "colorkey"],
                   help="filtro a usar; 'auto' testa os dois")
    p.add_argument("--similarity", default="auto",
                   help="tolerancia 0-1, ou 'auto' (omissao)")
    p.add_argument("--blend", type=float, default=0.10, help="suavizacao da margem")
    p.add_argument("--despill-type", default="auto", choices=["auto", "green", "blue"],
                   help="canal a limpar; 'auto' decide pela cor detectada")
    p.add_argument("--despill-mix", type=float, default=0.70)
    p.add_argument("--despill-expand", type=float, default=0.20)

    p.add_argument("--crf", type=int, default=26, help="qualidade inicial (menor = melhor)")
    p.add_argument("--max-mb", type=float, default=None,
                   help="se definido, sobe o crf ate caber neste tamanho")
    p.add_argument("--fps", type=float, default=None, help="forcar framerate")
    p.add_argument("--scale", default=None, help="redimensionar, ex. 720:-2")
    p.add_argument("--layout", default="h", choices=["h", "v"],
                   help="packed: cor/mascara lado a lado (h) ou empilhadas (v)")
    p.add_argument("--bg-color", default="0xFFFFFF", help="modo baked: cor de fundo")
    p.add_argument("--erode", type=int, default=2,
                   help="packed: passagens de erosao do alpha para matar o halo (0 desliga)")
    p.add_argument("--no-fill", action="store_true",
                   help="packed: nao preencher o fundo do plano de cor com o tom do sujeito")

    p.add_argument("--samples", type=int, default=4,
                   help="frames de amostra usados na deteccao e afinacao")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    exigir_ferramentas()

    v = sondar(args.entrada)
    print(f"entrada: {v.largura}x{v.altura} {v.fps:g}fps {v.duracao:.2f}s "
          f"{v.codec} {v.pix_fmt} ({v.n_frames} frames)")

    precisa_amostras = args.color == "auto" or args.key == "auto" or args.similarity == "auto"
    tmp = tempfile.mkdtemp(prefix="ckpack_") if precisa_amostras else None
    amostras: list[str] = []
    lg = 240
    at = max(2, int(240 * v.altura / v.largura))
    at -= at % 2

    if precisa_amostras:
        assert tmp is not None
        print("a extrair frames de amostra...")
        amostras = extrair_amostras(v, args.samples, lg, tmp)
        print(f"  {len(amostras)} frames a {lg}x{at}")

    if args.color == "auto":
        print("a detectar a cor do fundo...")
        cor = detectar_cor(amostras, lg, at)
        print(f"  cor detectada: {rgb_para_hex(cor)} {cor}")
    else:
        cor = hex_para_rgb(args.color)

    despill = tipo_despill(cor) if args.despill_type == "auto" else args.despill_type
    print(f"  despill: {despill}")

    if args.key == "auto" or args.similarity == "auto":
        tipos = ["chromakey", "colorkey"] if args.key == "auto" else [args.key]
        print("a afinar o key..." + (" (--verbose mostra a varredura)" if not args.verbose else ""))
        tipo, sim = afinar(amostras, cor, tipos, args.blend, args.verbose, despill, lg, at)
        if args.similarity != "auto":
            sim = float(args.similarity)
        if args.key != "auto":
            tipo = args.key
    else:
        tipo, sim = args.key, float(args.similarity)
        print(f"  key manual: {tipo} similarity={sim:.2f}")

    chave = cadeia_key(cor, tipo, sim, args.blend,
                       args.despill_mix, args.despill_expand, args.fps, args.scale,
                       despill)
    if args.verbose:
        print(f"  cadeia: {chave}")

    destino = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)

    fill = None
    if args.mode == "packed" and not args.no_fill:
        if amostras:
            fill = cor_do_sujeito(amostras, cor, tipo, sim, args.blend, lg, at)
        else:
            tmp2 = tempfile.mkdtemp(prefix="ckfill_")
            try:
                fill = cor_do_sujeito(extrair_amostras(v, args.samples, lg, tmp2),
                                      cor, tipo, sim, args.blend, lg, at)
            finally:
                shutil.rmtree(tmp2, ignore_errors=True)
        print(f"  preenchimento do fundo: {rgb_para_hex(fill)}")

    crf = args.crf
    bg = hex_para_rgb(args.bg_color)
    print(f"a codificar ({args.mode})...")
    codificar(v, destino, args.mode, chave, crf, bg, args.layout, args.verbose,
              args.erode, fill)
    print(f"  {mb(destino):.2f} MB  crf={crf}")

    if args.max_mb:
        limite = 51 if args.mode in ("packed", "baked") else 63
        while mb(destino) > args.max_mb and crf < limite:
            crf += 3
            print(f"  acima de {args.max_mb} MB, a repetir com crf={crf}...")
            codificar(v, destino, args.mode, chave, crf, bg, args.layout, args.verbose,
                      args.erode, fill)
            print(f"  {mb(destino):.2f} MB  crf={crf}")
        if mb(destino) > args.max_mb:
            print(f"  AVISO: nao cheguei a {args.max_mb} MB nem com crf={crf}. "
                  f"Reduz a resolucao (--scale) ou os fps (--fps).")

    if args.mode == "packed":
        print("a verificar o resultado...")
        verificar_packed(destino, args.layout)

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nescrito: {destino}  ({mb(destino):.2f} MB)")
    if args.mode == "packed":
        eixo = "lado a lado" if args.layout == "h" else "empilhadas"
        print(f"cor e mascara {eixo}; recompoe no cliente para obter transparencia.")


if __name__ == "__main__":
    main()