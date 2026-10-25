import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import Anthropic from "npm:@anthropic-ai/sdk@0.133.0";
import { createClient } from "npm:@supabase/supabase-js@2.45.4";
import nodemailer from "npm:nodemailer@6.9.15";

// All AI work for Paola's job app runs here, billed to the owner's Anthropic API key.
const MODEL = "claude-opus-5-5";
const PRICE = { input: 4 / 1e6, output: 20 / 1e6, search: 10 / 1000 };
const MONTHLY_CAP_USD = Number(Deno.env.get("MONTHLY_CAP_USD") ?? "25");
const MIN_SAVE_SCORE = 40;

const cors = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-cron-secret",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { ...cors, "Content-Type": "application/json" } });

const db = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
const anthropic = new Anthropic();

class UserError extends Error {}

async function authorize(req: Request): Promise<"user" | "cron" | null> {
  const cronSecret = req.headers.get("x-cron-secret");
  if (cronSecret) {
    const { data } = await db.from("app_config").select("value").eq("key", "cron_secret").single();
    return data?.value === cronSecret ? "cron" : null;
  }
  const token = (req.headers.get("authorization") ?? "").replace(/^Bearer\s+/i, "");
  if (!token) return null;
  const { data: { user } } = await db.auth.getUser(token);
  if (!user?.email) return null;
  const { data } = await db.from("allowed_emails").select("email").eq("email", user.email.toLowerCase()).maybeSingle();
  return data ? "user" : null;
}

async function spentThisMonth(): Promise<number> {
  const start = new Date();
  start.setUTCDate(1);
  start.setUTCHours(0, 0, 0, 0);
  const { data } = await db.from("ai_usage").select("cost_usd").gte("at", start.toISOString());
  return (data ?? []).reduce((s, r) => s + Number(r.cost_usd ?? 0), 0);
}

type AskOpts = { maxTokens?: number; effort?: "low" | "medium" | "high"; tools?: unknown[] };

async function ask(action: string, prompt: string, opts: AskOpts = {}): Promise<string> {
  const messages: Anthropic.Beta.BetaMessageParam[] = [{ role: "user", content: prompt }];
  let text = "";
  let usageIn = 0, usageOut = 0, searches = 0;
  try {
    // pause_turn: the server-side tool loop hit its iteration limit; resend to let it continue.
    for (let round = 0; round < 4; round++) {
      const params = {
        model: MODEL,
        max_tokens: opts.maxTokens ?? 8000,
        output_config: { effort: opts.effort ?? "medium" },
        betas: ["server-side-fallback-2026-07-01"],
        fallbacks: "default",
        messages,
        ...(opts.tools ? { tools: opts.tools } : {}),
      };
      // deno-lint-ignore no-explicit-any
      const res = await anthropic.beta.messages.create(params as any);
      usageIn += (res.usage.input_tokens ?? 0) + (res.usage.cache_read_input_tokens ?? 0) + (res.usage.cache_creation_input_tokens ?? 0);
      usageOut += res.usage.output_tokens ?? 0;
      // deno-lint-ignore no-explicit-any
      searches += (res.usage as any).server_tool_use?.web_search_requests ?? 0;
      if (res.stop_reason === "refusal") throw new UserError("Claude no ha querido procesar este texto.");
      if (res.stop_reason === "pause_turn") {
        messages.push({ role: "assistant", content: res.content });
        continue;
      }
      text = res.content.flatMap((b) => (b.type === "text" ? [b.text] : [])).join("");
      break;
    }
  } catch (e) {
    if (e instanceof UserError) throw e;
    if (e instanceof Anthropic.AuthenticationError) throw new UserError("La clave de la API no es válida. Revísala en Supabase.");
    if (e instanceof Anthropic.RateLimitError) throw new UserError("La API está saturada. Prueba en un minuto.");
    if (e instanceof Anthropic.APIError && e.status === 400 && /credit/i.test(e.message)) throw new UserError("Se han acabado los créditos de la API.");
    console.error(e);
    throw new UserError("No ha funcionado. Prueba otra vez en un momento.");
  } finally {
    const cost = usageIn * PRICE.input + usageOut * PRICE.output + searches * PRICE.search;
    if (usageIn || usageOut) {
      await db.from("ai_usage").insert({ action, model: MODEL, input_tokens: usageIn, output_tokens: usageOut, web_searches: searches, cost_usd: cost.toFixed(4) });
    }
  }
  return text.trim();
}

function parseJson<T>(text: string): T {
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = fence ? fence[1] : text;
  const start = body.search(/[[{]/);
  const end = Math.max(body.lastIndexOf("}"), body.lastIndexOf("]"));
  if (start < 0 || end < start) throw new UserError("La respuesta llegó incompleta. Prueba otra vez.");
  return JSON.parse(body.slice(start, end + 1)) as T;
}

async function getDoc(key: string) {
  const { data } = await db.from("docs").select("data").eq("key", key).maybeSingle();
  return (data?.data ?? {}) as Record<string, unknown>;
}
const cvForPrompt = (cv: Record<string, unknown>) => ({ ...cv, photo: undefined });

const JOB_FIELDS = `"title":"original title","company":"…","location":"City, Bundesland","accommodation":true|false|null,"contract":"e.g. Vollzeit, Saison Dez–Apr"|null,"salary":"as stated"|null,"german_level":"as stated"|null,"summary":"2 sentences in Spanish"`;
const SCORING = (prefs: Record<string, unknown>) =>
  `Score fit 0–100 for this candidate. Weigh role match, her real experience, German level required vs hers, staff accommodation (her preference: "${prefs.accommodation ?? "obligatorio"}"), location, contract type, start date. Be strict: 80+ only for a clear fit. If her CV is empty, score on her preferences only. "why" is one sentence in Spanish addressed to her as «tú».`;

async function translate(body: { lang: string; source: unknown }) {
  const lang = body.lang === "en" ? "English" : "German";
  const text = await ask("translate", `You translate a hospitality worker's CV from Spanish into ${lang} for job applications in Germany.
Use the standard ${lang} terms of the German hotel and restaurant trade (e.g. "camarera de sala" → ${lang === "German" ? '"Servicekraft"' : '"waitress"'}; "pisos" → "Housekeeping").
Polish the wording so it reads like a well-written CV: short, active, concrete. Each line of "tasks" becomes one concise line (keep "\\n" between lines). Never add facts, numbers or skills that are not in the source. Keep empty strings empty.
Reply with ONLY the same JSON object, same keys and same array lengths, values translated.

${JSON.stringify(body.source)}`, { effort: "low" });
  return { data: parseJson(text) };
}

async function letter(body: { jobId: string }) {
  const [{ data: job }, cv, prefs] = await Promise.all([
    db.from("jobs").select("*").eq("id", body.jobId).single(),
    getDoc("cv"),
    getDoc("prefs"),
  ]);
  if (!job) throw new UserError("No encuentro esa oferta.");
  const text = await ask("letter", `Write a short German cover letter (Anschreiben) for this hotel/restaurant job in Germany. Use the candidate's real data only; never invent experience, certificates or language levels. Formal but warm "Sie" tone, as German hospitality employers expect.
Structure: "Betreff: Bewerbung als …" line, "Sehr geehrte Damen und Herren," (or the contact person if given), 3 short paragraphs (why this house and job, relevant experience, availability date and interest in the staff accommodation if the job offers it, honest German level), "Mit freundlichen Grüßen" and her name. Max 220 words. Plain text, no markdown.

CANDIDATE CV (Spanish; "tr.de" may hold a German translation):
${JSON.stringify(cvForPrompt(cv))}

WHAT SHE WANTS:
${JSON.stringify(prefs)}

JOB:
${JSON.stringify({ title: job.title, company: job.company, location: job.location, contract: job.contract, accommodation: job.accommodation, german_level: job.german_level, summary: job.summary, ad_text: job.ad_text })}`, { effort: "medium" });
  await db.from("jobs").update({ letter: text, err: null }).eq("id", job.id);
  return { letter: text };
}

async function score() {
  const [{ data: todo }, cv, prefs] = await Promise.all([
    db.from("jobs").select("id,title,company,location,accommodation,contract,german_level,summary").is("score", null).neq("status", "descartada").limit(15),
    getDoc("cv"),
    getDoc("prefs"),
  ]);
  if (!todo?.length) return { scored: 0 };
  const text = await ask("score", `${SCORING(prefs)}
Reply with ONLY a JSON array: [{"id":"…","score":72,"why":"…"}]

CV: ${JSON.stringify({ ...cvForPrompt(cv), tr: undefined })}
PREFERENCES: ${JSON.stringify(prefs)}
JOBS: ${JSON.stringify(todo)}`, { effort: "low" });
  const rows = parseJson<{ id: string; score: number; why: string }[]>(text);
  let n = 0;
  for (const r of rows) {
    if (!todo.some((j) => j.id === r.id) || typeof r.score !== "number") continue;
    await db.from("jobs").update({ score: Math.round(r.score), why: String(r.why ?? "") }).eq("id", r.id);
    n++;
  }
  return { scored: n };
}

async function analyze(body: { url?: string; text: string }) {
  if (!body.text || body.text.length < 80) throw new UserError("Pega el texto completo de la oferta.");
  const [cv, prefs] = await Promise.all([getDoc("cv"), getDoc("prefs")]);
  const out = await ask("analyze", `Extract this German hospitality job ad and score it for the candidate. ${SCORING(prefs)}
Reply with ONLY JSON: {${JOB_FIELDS},"score":0-100,"why":"…"}

CV: ${JSON.stringify({ ...cvForPrompt(cv), tr: undefined })}
PREFERENCES: ${JSON.stringify(prefs)}
AD:
${body.text.slice(0, 30000)}`, { effort: "low" });
  const j = parseJson<Record<string, unknown>>(out);
  const row = {
    title: String(j.title ?? "Oferta"), company: j.company ?? null, location: j.location ?? null,
    url: body.url || null, source: "Añadida por ti", accommodation: j.accommodation ?? null,
    contract: j.contract ?? null, salary: j.salary ?? null, german_level: j.german_level ?? null,
    summary: j.summary ?? null, ad_text: body.text.slice(0, 30000),
    score: typeof j.score === "number" ? Math.round(j.score) : null, why: j.why ?? null, status: "interesa",
  };
  const { error } = await db.from("jobs").insert(row);
  if (error) throw new UserError(error.code === "23505" ? "Esa oferta ya está en tu lista." : "No se ha podido guardar la oferta.");
  return { ok: true };
}

async function getConfig(key: string): Promise<string> {
  const { data } = await db.from("app_config").select("value").eq("key", key).maybeSingle();
  return data?.value ?? "";
}

const escHtml = (s: unknown) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

// Notices go to the app's own Gmail address. Resend (signed up with that address) needs no domain
// to mail its owner; otherwise Gmail SMTP on port 465, since Supabase blocks outbound 25/587.
async function sendMail(subject: string, html: string) {
  const user = Deno.env.get("GMAIL_USER") || await getConfig("gmail_user"), pass = Deno.env.get("GMAIL_APP_PASSWORD");
  const resendKey = Deno.env.get("RESEND_API_KEY") || await getConfig("resend_api_key");
  if (resendKey && user) {
    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { Authorization: `Bearer ${resendKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ from: "Bewerbungsmappe <onboarding@resend.dev>", to: [user], subject, html }),
    });
    if (!res.ok) {
      console.error("resend", res.status, await res.text());
      throw new UserError("No se ha podido enviar el aviso por correo.");
    }
    return;
  }
  if (!user || !pass) throw new UserError("Falta configurar la cuenta de correo de los avisos.");
  const transport = nodemailer.createTransport({ host: "smtp.gmail.com", port: 465, secure: true, auth: { user, pass } });
  await transport.sendMail({ from: `Bewerbungsmappe <${user}>`, to: user, subject, html });
}

// Emails every good job found since the last notice, then marks them as notified.
async function notifyNewJobs(): Promise<number> {
  const min = Number(await getConfig("notify_min_score") || "70");
  const { data: fresh } = await db.from("jobs").select("id,title,company,location,url,score,why,accommodation,salary")
    .is("notified_at", null).gte("score", min).neq("status", "descartada").order("score", { ascending: false });
  if (!fresh?.length) return 0;
  const appUrl = await getConfig("app_url");
  const best = fresh[0];
  const subject = fresh.length === 1
    ? `Nueva oferta para ti: ${best.title} · ${best.location ?? ""} (${best.score})`
    : `${fresh.length} ofertas nuevas para ti · la mejor: ${best.title} (${best.score})`;
  const items = fresh.map((j) => `
    <tr><td style="padding:14px 0;border-top:1px solid #e3e7eb">
      <div style="font:700 16px/1.3 system-ui,sans-serif;color:#17212b">${escHtml(j.title)}</div>
      <div style="font:14px/1.4 system-ui,sans-serif;color:#5a6673">${escHtml(j.company)} · ${escHtml(j.location)}${j.salary ? " · " + escHtml(j.salary) : ""}</div>
      <div style="font:14px/1.5 system-ui,sans-serif;color:#17212b;margin-top:6px"><b style="color:#2c7a4c">${j.score}/100</b>${j.accommodation ? " · con alojamiento" : ""} — ${escHtml(j.why)}</div>
      ${j.url ? `<a href="${escHtml(j.url)}" style="display:inline-block;margin-top:8px;font:600 14px system-ui,sans-serif;color:#1f4e5f">Ver la oferta →</a>` : ""}
    </td></tr>`).join("");
  const html = `<div style="max-width:560px;margin:0 auto;padding:16px">
    <div style="font:600 12px system-ui,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:#8e6a22">Bewerbungsmappe</div>
    <h1 style="font:400 26px/1.2 Georgia,serif;color:#17212b;margin:6px 0 4px">${fresh.length === 1 ? "Hay una oferta que te encaja" : `Hay ${fresh.length} ofertas que te encajan`}</h1>
    <p style="font:14px/1.5 system-ui,sans-serif;color:#5a6673;margin:0 0 8px">Puntuadas con tu CV y lo que buscas. En la app puedes pedir la carta en alemán.</p>
    <table role="presentation" style="width:100%;border-collapse:collapse">${items}</table>
    ${appUrl ? `<a href="${escHtml(appUrl)}#ofertas" style="display:inline-block;margin-top:16px;padding:10px 16px;border-radius:8px;background:#1f4e5f;color:#fff;font:600 14px system-ui,sans-serif;text-decoration:none">Abrir mis ofertas</a>` : ""}
  </div>`;
  await sendMail(subject, html);
  await db.from("jobs").update({ notified_at: new Date().toISOString() }).in("id", fresh.map((j) => j.id));
  return fresh.length;
}

// Bundesagentur für Arbeit public job API: v6 for search, v4 for the full ad text.
const BA = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service";
const HOUSING = /unterkunft|personalzimmer|personalwohn|mitarbeiterwohn|mitarbeiterzimmer|wohnm[öo]glichkeit|kost und logis|zimmer (wird|kann) (gestellt|zur verf)/i;

async function baGet(path: string) {
  const res = await fetch(BA + path, { headers: { "X-API-Key": "jobboerse-jobsuche" } });
  if (!res.ok) throw new Error(`BA ${res.status} ${path}`);
  return res.json();
}

type Prefs = Record<string, unknown>;
type Row = Record<string, unknown>;

async function searchArbeitsagentur(prefs: Prefs, cv: Record<string, unknown>, known: Set<string>): Promise<Row[]> {
  const roles = (prefs.roles as string[] | undefined)?.length ? prefs.roles as string[] : ["Servicekraft", "Housekeeping", "Rezeption"];
  const needHousing = prefs.accommodation === "obligatorio";
  const terms = needHousing || prefs.accommodation === "preferible" ? ["Unterkunft", "Personalzimmer"] : [""];
  const zones = String(prefs.zones ?? "").split(",").map((z) => z.trim()).filter((z) => z && !/toda|alemania|deutschland/i.test(z)).slice(0, 3);
  const queries = roles.flatMap((r) => terms.map((t) => `${r} ${t}`.trim()));
  const refs = new Map<string, Row>();
  await Promise.all(queries.flatMap((q) => (zones.length ? zones : [""]).map(async (wo) => {
    const qs = new URLSearchParams({ was: q, angebotsart: "1", size: "25", veroeffentlichtseit: "14" });
    if (wo) qs.set("wo", wo);
    try {
      const data = await baGet(`/pc/v6/jobs?${qs}`);
      for (const j of data.ergebnisliste ?? []) if (j.referenznummer) refs.set(j.referenznummer, j);
    } catch (e) { console.error(e); }
  })));
  const fresh = [...refs.values()].filter((j) => !known.has(`https://www.arbeitsagentur.de/jobsuche/jobdetail/${j.referenznummer}`)).slice(0, 30);
  const detailed: Row[] = [];
  for (let i = 0; i < fresh.length; i += 6) {
    await Promise.all(fresh.slice(i, i + 6).map(async (j) => {
      try {
        const d = await baGet(`/pc/v4/jobdetails/${btoa(String(j.referenznummer))}`);
        const text = `${d.stellenangebotsTitel ?? ""}\n${d.stellenangebotsBeschreibung ?? ""}`;
        if (needHousing && !HOUSING.test(text)) return;
        const loc = (d.stellenlokationen ?? j.stellenlokationen ?? [])[0]?.adresse ?? {};
        if (loc.land && loc.land !== "DEUTSCHLAND") return;
        detailed.push({
          ref: j.referenznummer, title: d.stellenangebotsTitel ?? j.stellenangebotsTitel, company: d.firma ?? j.firma,
          place: [loc.ort, loc.region].filter(Boolean).join(", "), posted: j.datumErsteVeroeffentlichung ?? null,
          contract: [d.arbeitszeitVollzeit ? "Vollzeit" : null, d.vertragsdauer === "BEFRISTET" ? `befristet ${d.befristungInMonaten ?? ""} Monate` : d.vertragsdauer === "UNBEFRISTET" ? "unbefristet" : null].filter(Boolean).join(", "),
          pay: d.festgehalt ? `${d.festgehalt} € (${d.verguetungsangabe})` : null,
          ad: String(d.stellenangebotsBeschreibung ?? "").slice(0, 1800),
        });
      } catch (e) { console.error(e); }
    }));
  }
  if (!detailed.length) return [];
  const out = await ask("search-ba", `These are real, open job ads from the German federal job agency. Extract and score each one for the candidate. ${SCORING(prefs)}
Reply with ONLY a JSON array, one object per ad you consider relevant to her (drop clearly unrelated ones): [{"ref":"…",${JOB_FIELDS},"score":0-100,"why":"…"}]. Keep "title" and "company" as in the ad; "location" as "City, Bundesland".

CV: ${JSON.stringify({ ...cvForPrompt(cv), tr: undefined })}
PREFERENCES: ${JSON.stringify(prefs)}
ADS: ${JSON.stringify(detailed)}`, { effort: "low", maxTokens: 12000 });
  const scored = parseJson<Row[]>(out);
  return (Array.isArray(scored) ? scored : []).filter((s) => detailed.some((d) => d.ref === s.ref)).map((s) => {
    const d = detailed.find((x) => x.ref === s.ref)!;
    return {
      ...s, url: `https://www.arbeitsagentur.de/jobsuche/jobdetail/${s.ref}`, source: "Arbeitsagentur",
      posted: d.posted, contract: s.contract ?? d.contract ?? null, salary: s.salary ?? d.pay ?? null,
    };
  });
}

async function searchWeb(prefs: Prefs, cv: Record<string, unknown>, known: Set<string>, manual: boolean): Promise<Row[]> {
  const roles = (prefs.roles as string[] | undefined)?.length ? prefs.roles as string[] : ["Servicekraft", "Housekeeping", "Rezeption"];
  const focus = manual ? `any of: ${roles.join(", ")}` : `${roles[new Date().getUTCDate() % roles.length]} (other roles she accepts: ${roles.join(", ")})`;
  const text = await ask(manual ? "search-manual" : "search", `Find CURRENTLY OPEN hospitality jobs in Germany for this candidate and return the best new ones.
Today is ${new Date().toISOString().slice(0, 10)}. Focus on: ${focus}. Zones: ${prefs.zones || "all of Germany"}. Keywords: ${prefs.keywords || "-"}. Exclude: ${prefs.exclude || "-"}. Contract: ${(prefs.contract as string[] | undefined)?.join(", ") || "any"}. Can start: ${prefs.startFrom || "soon"}. Minimum salary: ${prefs.salaryMin || "not set"} € gross/month.
Accommodation: ${prefs.accommodation === "obligatorio" ? "ONLY keep listings that explicitly offer staff housing (Personalunterkunft, Mitarbeiterwohnung, Unterkunft möglich, Kost und Logis)" : prefs.accommodation === "preferible" ? "prefer listings with staff housing" : "not required"}.
Search job boards such as hogapage.de, hotelcareer.de, rollingpin.de, stepstone.de, indeed.de and hotels' own career pages, in German (e.g. "Servicekraft mit Unterkunft", "Rezeption Personalzimmer", "Housekeeping Kost und Logis"). The federal Arbeitsagentur board is covered separately; skip it. Every job must have the direct URL of one specific listing (not a search-results page) taken from your search or fetch results. Open the most promising listings with web_fetch to confirm details when you can. Never invent URLs or details. Aim for 3–6 good jobs.
Skip anything already known: ${JSON.stringify([...known].slice(0, 300))}
${SCORING(prefs)}
CV: ${JSON.stringify({ ...cvForPrompt(cv), tr: undefined })}

Finish with ONLY a JSON array of at most 6 jobs: [{${JOB_FIELDS},"url":"…","source":"site name","posted":"YYYY-MM-DD"|null,"score":0-100,"why":"…"}]. Return [] if nothing qualifies.`, {
    effort: "medium",
    maxTokens: 16000,
    tools: [
      { type: "web_search_20260209", name: "web_search", max_uses: 8, user_location: { type: "approximate", country: "DE" } },
      { type: "web_fetch_20260209", name: "web_fetch", max_uses: 10 },
    ],
  });
  const rows = parseJson<Row[]>(text);
  return Array.isArray(rows) ? rows : [];
}

async function searchBusy(): Promise<boolean> {
  const meta = await getDoc("meta");
  return Boolean(meta.searching && Date.now() - Date.parse(String(meta.searching)) < 5 * 60_000);
}

// Two sources at once: the Arbeitsagentur API (real ads, read in full) and Claude's web search
// for the private boards. Daily web runs focus on one role; manual runs cover every role.
async function search(manual = false) {
  const meta = await getDoc("meta");
  await db.from("docs").upsert({ key: "meta", data: { ...meta, searching: new Date().toISOString(), error: null } });
  let found = 0, error: string | null = null, debug = "";
  try {
    const [cv, prefs, { data: existing }] = await Promise.all([
      getDoc("cv"), getDoc("prefs"),
      db.from("jobs").select("url").order("added_at", { ascending: false }).limit(1000),
    ]);
    const known = new Set((existing ?? []).map((e) => String(e.url)).filter(Boolean));
    // Web search found nothing in testing at ~$1.50 a run, so it stays off unless app_config.web_search = "on".
    const webOn = (await getConfig("web_search")) === "on";
    const [ba, web] = await Promise.allSettled([
      searchArbeitsagentur(prefs, cv, known),
      webOn ? searchWeb(prefs, cv, known, manual) : Promise.resolve([] as Row[]),
    ]);
    const rows: Row[] = [];
    for (const [name, r] of [["arbeitsagentur", ba], ["web", web]] as const) {
      if (r.status === "fulfilled") { rows.push(...r.value); debug += `${name}: ${r.value.length}\n`; }
      else { debug += `${name} failed: ${r.reason instanceof Error ? r.reason.message : String(r.reason)}\n`; console.error(r.reason); }
    }
    if (ba.status === "rejected" && web.status === "rejected") {
      error = web.reason instanceof UserError ? web.reason.message : "La búsqueda ha fallado.";
    }
    for (const j of rows) {
      if (!j.url || !j.title || known.has(String(j.url))) continue;
      if (typeof j.score === "number" && j.score < MIN_SAVE_SCORE) continue;
      const { error: e } = await db.from("jobs").insert({
        title: String(j.title), company: j.company ?? null, location: j.location ?? null, url: String(j.url),
        source: j.source ?? null, posted: j.posted ?? null, accommodation: j.accommodation ?? null,
        contract: j.contract ?? null, salary: j.salary ?? null, german_level: j.german_level ?? null,
        summary: j.summary ?? null, score: typeof j.score === "number" ? Math.round(j.score) : null, why: j.why ?? null,
      });
      if (!e) { found++; known.add(String(j.url)); }
      else debug += `insert ${j.url}: ${e.message}\n`;
    }
  } catch (e) {
    error = e instanceof UserError ? e.message : "La búsqueda ha fallado.";
    console.error(e);
  }
  try {
    await notifyNewJobs();
  } catch (e) {
    console.error(e);
    error = error ?? (e instanceof UserError ? e.message : "No se ha podido enviar el aviso por correo.");
  }
  const fresh = await getDoc("meta");
  await db.from("docs").upsert({ key: "meta", data: { ...fresh, searching: null, lastSearch: new Date().toISOString(), found, error, debug: debug.slice(0, 2000) } });
  return { found, error };
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors });
  const who = await authorize(req);
  if (!who) return json({ error: "No tienes acceso." }, 401);
  let body: Record<string, unknown> = {};
  try { body = await req.json(); } catch { /* empty body */ }
  const action = who === "cron" ? "search" : String(body.action ?? "");
  try {
    if ((await spentThisMonth()) >= MONTHLY_CAP_USD) {
      return json({ error: `Se ha llegado al límite de gasto de este mes (${MONTHLY_CAP_USD} $).` }, 402);
    }
    switch (action) {
      case "translate": return json(await translate(body as { lang: string; source: unknown }));
      case "letter": return json(await letter(body as { jobId: string }));
      case "score": return json(await score());
      case "analyze": return json(await analyze(body as { url?: string; text: string }));
      case "test-email":
        await sendMail("Prueba de avisos · Bewerbungsmappe", `<p style="font:15px system-ui,sans-serif">Los avisos de ofertas nuevas llegarán a este correo.</p>`);
        return json({ ok: true });
      case "search":
        if (await searchBusy()) return json({ busy: true });
        // Searching takes minutes; answer now and keep working in the background.
        EdgeRuntime.waitUntil(search(body.manual === true));
        return json({ started: true });
      default: return json({ error: "Acción desconocida." }, 400);
    }
  } catch (e) {
    if (e instanceof UserError) return json({ error: e.message }, 400);
    console.error(e);
    return json({ error: "Error inesperado." }, 500);
  }
});
