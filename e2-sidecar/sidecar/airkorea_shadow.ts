/**
 * airkorea_shadow.ts — AirKorea shadow collector for the Oracle E2.1.Micro sidecar.
 *
 * Wave 1 of plan `polymorphic-kindling-pebble.md`. This is a SHADOW collector:
 * it writes only its own output file, which is published to HF under
 * `airkorea-shadow/`. It never touches the canonical Supabase tables or the live
 * HF paths, so it can run alongside the GitHub Actions collector without
 * affecting it.
 *
 * What Wave 1 measures: whether a systemd timer closes the hourly gaps GitHub
 * cron drift opens. `airkorea-collect.yml:5-8` records 5~8 hours lost per day,
 * and the AirKorea realtime API has no backfill, so a missed hour is gone for
 * good. The output carries `collection_hour_kst` precisely so shadow and
 * canonical hour coverage can be diffed.
 *
 * Row parsing, station coordinates and the KST timestamp rule are imported from
 * the canonical Edge Function modules rather than reimplemented — a shadow feed
 * built on different parsing would not be comparable. Only the response
 * envelope schema is redeclared: the Edge Function's index.ts runs Deno.serve at
 * import time, so it cannot be imported at all.
 *
 * Layout (both on E2 and locally):
 *   <root>/repo/      git clone of AirLens-cloud/airlens-ml (2026-10-21까지 AirLens)
 *   <root>/sidecar/   this file
 *
 * Run:
 *   deno run --allow-read=<root> --allow-write=<out> \
 *     --allow-env=AIRKOREA_API_KEY \
 *     --allow-net=apis.data.go.kr,esm.sh \
 *     --lock=deno.lock \
 *     airkorea_shadow.ts --out /var/lib/airlens/shadow
 */
import { z } from 'https://esm.sh/zod@3.23.8'
import { filterValidItems, type AirkoreaItem } from '../repo/apps/web/supabase/functions/airkorea-collector/itemFilter.ts'
import {
  fetchStationCoordMap,
  coordMapKey,
  type StationCoord,
} from '../repo/apps/web/supabase/functions/airkorea-collector/stationMeta.ts'
import { parseDataTimeKst } from '../repo/apps/web/supabase/functions/airkorea-collector/dataTime.ts'

const AIRKOREA_BASE =
  'https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty'

const SIDOS = [
  '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
  '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
]

const MAX_PAGES = 5

// Envelope validation mirrors the canonical collector's boundary: an envelope
// shape change is real schema drift and must throw, while a single broken row is
// dropped by filterValidItems. That is why items stays z.unknown() here.
const AirkoreaResponseSchema = z.object({
  response: z.object({
    header: z.object({
      resultCode: z.string().optional(),
      resultMsg: z.string().optional(),
    }).passthrough().optional(),
    body: z.object({
      totalCount: z.number().optional(),
      items: z.array(z.unknown()).optional(),
      pageNo: z.number().optional(),
      numOfRows: z.number().optional(),
    }).passthrough().optional(),
  }).passthrough(),
}).passthrough()

function argValue(flag: string, fallback: string): string {
  const i = Deno.args.indexOf(flag)
  return i >= 0 && Deno.args[i + 1] ? Deno.args[i + 1] : fallback
}

async function fetchSidoPage(
  apiKey: string,
  sido: string,
  pageNo: number,
): Promise<{ items: AirkoreaItem[]; skipped: number; totalCount: number }> {
  const params = new URLSearchParams({
    serviceKey: apiKey,
    returnType: 'json',
    sidoName: sido,
    numOfRows: '100',
    pageNo: String(pageNo),
    ver: '1.0',
  })

  const res = await fetch(`${AIRKOREA_BASE}?${params}`, {
    signal: AbortSignal.timeout(10_000),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`AirKorea ${res.status}: ${text.slice(0, 200)}`)
  }

  const parsed = AirkoreaResponseSchema.safeParse(await res.json())
  if (!parsed.success) {
    throw new Error(
      `AirKorea schema drift for ${sido}: ${parsed.error.issues[0]?.message ?? 'unknown'}`,
    )
  }

  const resultCode = parsed.data.response?.header?.resultCode
  if (resultCode && resultCode !== '00') {
    throw new Error(
      `AirKorea API error for ${sido}: ${parsed.data.response?.header?.resultMsg ?? resultCode}`,
    )
  }

  const { items, skipped } = filterValidItems(parsed.data.response?.body?.items ?? [])
  return { items, skipped, totalCount: parsed.data.response?.body?.totalCount ?? 0 }
}

async function fetchAllPages(apiKey: string, sido: string) {
  const first = await fetchSidoPage(apiKey, sido, 1)
  const items = [...first.items]
  let skipped = first.skipped

  const pages = Math.min(Math.ceil(first.totalCount / 100), MAX_PAGES)
  for (let page = 2; page <= pages; page++) {
    const next = await fetchSidoPage(apiKey, sido, page)
    items.push(...next.items)
    skipped += next.skipped
  }
  return { items, skipped }
}

const apiKey = Deno.env.get('AIRKOREA_API_KEY')
if (!apiKey) {
  console.error(JSON.stringify({ event: 'airkorea-shadow:missing_api_key' }))
  Deno.exit(1)
}

const outDir = argValue('--out', '/var/lib/airlens/shadow')
const startedAt = new Date()

// Station coordinates come from the same metadata service the canonical
// collector uses. A failure here degrades to null coordinates rather than
// aborting — coordinates are never fabricated (stationMeta.ts:118-137).
let coordMap = new Map<string, StationCoord>()
try {
  coordMap = await fetchStationCoordMap(apiKey)
} catch (err) {
  console.warn(JSON.stringify({ event: 'airkorea-shadow:coord_map_failed', error: String(err) }))
}

const rows: Record<string, unknown>[] = []
const failedSidos: string[] = []
let skippedRows = 0

for (const sido of SIDOS) {
  try {
    const { items, skipped } = await fetchAllPages(apiKey, sido)
    skippedRows += skipped
    for (const item of items) {
      const coords = coordMap.get(coordMapKey(sido, item.stationName)) ?? null
      rows.push({
        station_name: item.stationName,
        sido_name: sido,
        mango_name: item.mangName ?? null,
        pm25: item.pm25Value ?? null,
        pm10: item.pm10Value ?? null,
        o3: item.o3Value ?? null,
        no2: item.no2Value ?? null,
        so2: item.so2Value ?? null,
        co: item.coValue ?? null,
        khai_value: item.khaiValue ?? null,
        khai_grade: item.khaiGrade ?? null,
        pm25_grade: item.pm25Grade ?? null,
        pm10_grade: item.pm10Grade ?? null,
        lat: coords?.lat ?? null,
        lon: coords?.lon ?? null,
        // Canonical KST normalization — the field the hour-coverage diff keys on.
        data_time: parseDataTimeKst(item.dataTime),
        data_time_raw: item.dataTime ?? null,
      })
    }
  } catch (err) {
    failedSidos.push(sido)
    console.error(JSON.stringify({ event: 'airkorea-shadow:sido_failed', sido, error: String(err) }))
  }
}

// Collection hour in KST — the unit Wave 1 compares against the canonical feed.
const kstNow = new Date(startedAt.getTime() + 9 * 60 * 60 * 1000)
const stamp = kstNow.toISOString().slice(0, 13).replace(/[-T]/g, '')

await Deno.mkdir(outDir, { recursive: true })
const outFile = `${outDir}/airkorea-${stamp}.json`
await Deno.writeTextFile(
  outFile,
  JSON.stringify({
    schema_version: '1.0.0',
    source: 'airkorea-shadow',
    collected_at: startedAt.toISOString(),
    collection_hour_kst: stamp,
    station_count: rows.length,
    failed_sidos: failedSidos,
    skipped_rows: skippedRows,
    coord_map_size: coordMap.size,
    rows,
  }),
)

console.log(JSON.stringify({
  event: 'airkorea-shadow:done',
  out: outFile,
  stations: rows.length,
  failed_sidos: failedSidos.length,
  skipped_rows: skippedRows,
  coord_map_size: coordMap.size,
}))

// A run that collected nothing is a failure, not a quiet success — the same
// silence-breaking rule airkorea-collect.yml:57 applies to history_rows.
if (rows.length === 0) {
  console.error(JSON.stringify({ event: 'airkorea-shadow:zero_rows' }))
  Deno.exit(1)
}
