/* 머리말의 철가루 무늬.
   헬름홀츠 코일의 단면(전류가 반대 방향인 도선 두 쌍)이 만드는 2차원 자기장을 계산해
   철가루 조각을 그 방향으로 눕힌다. 페이지가 열릴 때 한 번만 흩어진 상태에서 정렬된다. */

const AREA_PER_BIT = 62;            // 철가루 조각 하나가 차지하는 넓이(px²)
const ALIGN_MS = 900;

function rng(seed) {               // mulberry32: 새로고침해도 같은 무늬
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fieldAt(x, y, wires) {
  let bx = 0, by = 0;
  for (const w of wires) {
    const dx = x - w.x, dy = y - w.y, r2 = dx * dx + dy * dy + 40;
    bx += -w.i * dy / r2;
    by += w.i * dx / r2;
  }
  return [bx, by];
}

function build(W, H) {
  const small = W < 760;
  const cx = W * (small ? 0.5 : 0.74);
  const cy = H * 0.5;
  const gap = small ? 30 : 46;     // 코일 간격의 절반
  const R = H * 0.3;               // 코일 반지름
  const wires = [
    { x: cx - gap, y: cy - R, i: 1 }, { x: cx - gap, y: cy + R, i: -1 },
    { x: cx + gap, y: cy - R, i: 1 }, { x: cx + gap, y: cy + R, i: -1 },
  ];
  const rand = rng(1830);
  const clear = small ? 0 : W * 0.4;        // 넓은 화면에서는 왼쪽 글자 자리를 비운다
  const n = Math.round((W - clear * 0.7) * H / AREA_PER_BIT);
  const bits = [];
  for (let i = 0; i < n; i++) {
    const x = clear * 0.7 + rand() * (W - clear * 0.7), y = rand() * H;
    const near = wires.some(w => Math.hypot(x - w.x, y - w.y) < 7);
    if (near) continue;
    const [bx, by] = fieldAt(x, y, wires);
    const m = Math.hypot(bx, by);
    const fade = small ? 1 : Math.min(1, Math.max(0, (x - clear * 0.7) / (W * 0.24)));
    const strength = Math.min(1, m * 34);
    const alpha = fade * (0.1 + 0.62 * Math.sqrt(strength)) * (0.7 + 0.3 * rand());
    if (alpha < 0.03) continue;
    bits.push({ x, y, to: Math.atan2(by, bx), from: rand() * Math.PI, len: 3.5 + 5 * strength + rand() * 3, alpha });
  }
  return { bits, wires };
}

function paint(ctx, W, H, scene, rgb, k) {
  ctx.clearRect(0, 0, W, H);
  ctx.lineCap = 'round';
  ctx.lineWidth = 1.1;
  for (const b of scene.bits) {
    let d = b.to - b.from;
    d = Math.atan2(Math.sin(2 * d), Math.cos(2 * d)) / 2;   // 철가루는 앞뒤 구분이 없다
    const a = b.from + d * k;
    const dx = Math.cos(a) * b.len / 2, dy = Math.sin(a) * b.len / 2;
    ctx.strokeStyle = `rgba(${rgb},${b.alpha.toFixed(3)})`;
    ctx.beginPath();
    ctx.moveTo(b.x - dx, b.y - dy);
    ctx.lineTo(b.x + dx, b.y + dy);
    ctx.stroke();
  }
}

export function drawFilings(canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let played = false, W = 0, H = 0, scene = null;

  const color = () => getComputedStyle(document.documentElement).getPropertyValue('--filings').trim();
  function layout() {
    const box = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = box.width; H = box.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scene = build(W, H);
  }
  function play() {
    const rgb = color();
    if (still || played) { paint(ctx, W, H, scene, rgb, 1); return; }
    played = true;
    const t0 = performance.now();
    const step = now => {
      const t = Math.min(1, (now - t0) / ALIGN_MS);
      paint(ctx, W, H, scene, rgb, 1 - Math.pow(1 - t, 3));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  layout(); play();
  let timer = 0;
  addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => { layout(); play(); }, 120);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => play());
}
