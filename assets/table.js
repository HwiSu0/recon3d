/* Capstone CSV 읽기. 열 이름에 쉼표가 있어 머리글은 따옴표를 따져서 나눈다. */
export function parseHeader(line) {
  const cols = []; let cur = '', quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { cols.push(cur); cur = ''; }
    else cur += ch;
  }
  cols.push(cur);
  return cols;
}

/* 필요한 열만 숫자로 잘라 낸다. Run이 많은 파일도 파이썬이 표 전체를 읽지 않아도 된다.
   빈 칸은 NaN이고, 결과는 행 순서대로 펼친 Float64 배열이다. */
export async function readTable(file, columns) {
  const reader = file.stream().pipeThrough(new TextDecoderStream()).getReader();
  const nc = columns.length;
  const row = new Float64Array(nc);
  let data = new Float64Array(nc * 8192), rows = 0, rest = '', want = null, slot = null;

  const addHeader = line => {
    const names = parseHeader(line.replace(/^﻿/, ''));
    const where = columns.map(c => names.indexOf(c));
    const missing = columns.find((c, i) => where[i] < 0);
    if (missing) throw new Error(`CSV에 "${missing}" 열 없음`);
    const order = where.map((col, i) => [col, i]).sort((a, b) => a[0] - b[0]);
    want = order.map(x => x[0]); slot = order.map(x => x[1]);
  };

  const addRow = line => {
    row.fill(NaN);
    const quoted = line.includes('"') ? parseHeader(line) : null;
    let pos = 0, col = 0, k = 0, any = false;
    while (k < want.length) {
      let end = line.length, text;
      if (quoted) text = quoted[col] === undefined ? '' : quoted[col];
      else { end = line.indexOf(',', pos); if (end < 0) end = line.length; }
      if (col === want[k]) {
        if (!quoted) text = line.slice(pos, end);
        if (text !== '') { row[slot[k]] = Number(text); any = true; }
        k += 1;
      }
      if (quoted ? col + 1 >= quoted.length : end === line.length) break;
      pos = end + 1; col += 1;
    }
    if (!any) return;
    if ((rows + 1) * nc > data.length) { const grown = new Float64Array(data.length * 2); grown.set(data); data = grown; }
    data.set(row, rows * nc);
    rows += 1;
  };

  const addLine = line => {
    if (line.endsWith('\r')) line = line.slice(0, -1);
    if (want) addRow(line);
    else addHeader(line);
  };

  for (;;) {
    const { value, done } = await reader.read();
    const text = value ? rest + value : rest;
    let start = 0, end;
    while ((end = text.indexOf('\n', start)) >= 0) { addLine(text.slice(start, end)); start = end + 1; }
    rest = text.slice(start);
    if (done) break;
  }
  if (rest) addLine(rest);
  return { columns, buffer: data.buffer.slice(0, rows * nc * 8) };
}
