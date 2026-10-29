const SERVICE = 'http://127.0.0.1:8765';

// Desktop panel bridge — this extension copy is configured for Vivaldi.
const PANEL_URL = "http://127.0.0.1:8767";
const PANEL_BROWSER = "Chrome";

async function panelBlobToBase64(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000)
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}
async function publishToDesktopPanel(panelLatest, blob, tab) {
  try {
    if (!panelLatest || panelLatest.status !== "recognized" || !panelLatest.catchName) return;
    await fetch(`${PANEL_URL}/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        browser: PANEL_BROWSER,
        name: panelLatest.displayName || panelLatest.catchName || panelLatest.pokemon,
        command: panelLatest.command || "",
        score: panelLatest.score ?? null,
        image_base64: blob ? await panelBlobToBase64(blob) : "",
        image_mime: blob?.type || "image/png",
        image_url: panelLatest.imageUrl || "",
        tabId: tab?.id ?? panelLatest.sourceTabId ?? null,
        windowId: tab?.windowId ?? null
      })
    });
  } catch (e) {
    console.warn("[Pokémon Detector] Desktop panel bridge:", e);
  }
}
function startDesktopPanelFocusPolling() {
  const poll = async () => {
    try {
      const response = await fetch(`${PANEL_URL}/focus-poll?browser=${encodeURIComponent(PANEL_BROWSER)}`);
      const data = await response.json();
      const request = data?.request;
      if (request && Number.isInteger(request.tabId)) {
        if (Number.isInteger(request.windowId))
          await chrome.windows.update(request.windowId, { focused: true }).catch(() => {});
        await chrome.tabs.update(request.tabId, { active: true }).catch(() => {});
        await chrome.tabs.sendMessage(request.tabId, { type: "DESKTOP_PANEL_FOCUS_DISCORD" }).catch(() => {});
      }
    } catch (_) {}
  };
  poll();
  setInterval(poll, 1000);
}
startDesktopPanelFocusPolling();

let enabled = false;
let autoSwitch = true;
let outputMode = 'command'; // 'command' or 'name'
let languageMode = 'current'; // current, selected, all, random
let selectedLanguage = 'english';
let localizedDataPromise = null;
let userAltNamesPromise = null;
let japaneseDataPromise = null;
let poketwoCsvPromise = null;
const POKETWO_CSV_URL = 'https://huggingface.co/spaces/LightningDev1/PokeTwo/resolve/main/pokemon.csv';
const JAPANESE_NAMES_URL = 'https://unpkg.com/pokemon@3.3.0/data/ja.json';
let latest = { status: 'idle', pokemon: '', score: null, command: '', error: '' };
const seen = new Set();
let processingQueue = Promise.resolve();
const DISCORD_URL = /^https:\/\/(?:[^/]+\.)?discord\.com\//i;

function channelFromUrl(url) {
  const match = String(url || '').match(/^https:\/\/(?:[^/]+\.)?discord\.com\/channels\/(\d+)\/(\d+)/i);
  return match ? { guildId: match[1], channelId: match[2], url: `https://discord.com/channels/${match[1]}/${match[2]}` } : null;
}
async function loadSettings() {
  const state = await chrome.storage.local.get(['monitoringEnabled', 'autoSwitch', 'latest', 'selectedChannel', 'outputMode', 'languageMode', 'selectedLanguage']);
  enabled = !!state.monitoringEnabled;
  autoSwitch = state.autoSwitch !== false;
  outputMode = state.outputMode === 'name' ? 'name' : 'command';
  languageMode = ['current', 'selected', 'all', 'random'].includes(state.languageMode) ? state.languageMode : 'current';
  selectedLanguage = ['english', 'english_alternate', 'user_alt_names', 'german', 'french', 'japanese_romanized', 'japanese'].includes(state.selectedLanguage) ? state.selectedLanguage : 'english';
  latest = state.latest || latest;
}
async function getSelectedChannels() {
  const state = await chrome.storage.local.get(['selectedChannels', 'selectedChannel']);
  if (Array.isArray(state.selectedChannels)) return state.selectedChannels;
  // Migrate the previous single-channel setting without losing it.
  const legacy = state.selectedChannel ? [state.selectedChannel] : [];
  if (legacy.length) await chrome.storage.local.set({ selectedChannels: legacy });
  return legacy;
}
function isSelectedChannel(url, selectedChannels) {
  const current = channelFromUrl(url);
  return !!(current && Array.isArray(selectedChannels) && selectedChannels.some(selected =>
    selected && current.channelId === selected.channelId && current.guildId === selected.guildId
  ));
}

function stripLatinDiacritics(value) {
  return String(value || '').normalize('NFKD').replace(/\p{M}/gu, '').normalize('NFC').trim();
}
// Keep letters, numbers and spaces only: strips emoji, gender icons, accents and punctuation.
function cleanPokemonName(value) {
  return stripLatinDiacritics(value).replace(/[^\p{L}\p{N}\s]/gu, '').replace(/\s+/g, ' ').trim();
}
function normalizePlainName(value) {
  return cleanPokemonName(value).toLowerCase();
}
function normalizeBlacklistNames(values) {
  return [...new Set((Array.isArray(values) ? values : String(values || '').split(/[\n,;]+/))
    .map(value => normalizePlainName(value)).filter(Boolean))];
}
function isPokemonBlacklisted(names, candidates) {
  const blacklist = new Set(normalizeBlacklistNames(names));
  return (candidates || []).some(value => {
    const normalized = normalizePlainName(value);
    return normalized && blacklist.has(normalized);
  });
}
function normalizeSlug(value) {
  return normalizePlainName(value).replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}
function slugTokenKey(value) {
  return normalizeSlug(value).split('-').filter(Boolean).sort().join('-');
}
function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.some(cell => cell.trim())) rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field.length || row.length) { row.push(field); if (row.some(cell => cell.trim())) rows.push(row); }
  if (!rows.length) return [];
  const headers = rows.shift().map(h => h.trim());
  const keep = ['slug','dex_number','name.ja','name.ja_r','name.ja_t','name.en','name.en2','name.de','name.fr','is_form','form_item','evo.mega','evo.mega_x','evo.mega_y'];
  const indices = keep.map(key => headers.indexOf(key));
  return rows.map(cells => {
    const obj = {};
    keep.forEach((key, i) => { if (indices[i] >= 0) obj[key] = String(cells[indices[i]] || '').trim(); });
    return obj;
  }).filter(obj => obj.slug);
}
async function getPoketwoCsvRecords() {
  if (!poketwoCsvPromise) {
    poketwoCsvPromise = (async () => {
      const cached = await chrome.storage.local.get(['poketwoNameRecordsCache']);
      if (Array.isArray(cached.poketwoNameRecordsCache) && cached.poketwoNameRecordsCache.length > 500) return cached.poketwoNameRecordsCache;
      const response = await fetch(POKETWO_CSV_URL, { cache: 'force-cache' });
      if (!response.ok) throw new Error(`Pokétwo names source returned HTTP ${response.status}`);
      const records = parseCsv(await response.text());
      if (records.length < 500) throw new Error('Pokétwo names source returned an unexpected dataset');
      try { await chrome.storage.local.set({ poketwoNameRecordsCache: records, poketwoNameRecordsCacheAt: Date.now() }); } catch (e) { console.warn('[Pokémon Detector] Could not cache Pokétwo names:', e); }
      return records;
    })().catch(error => { poketwoCsvPromise = null; throw error; });
  }
  return poketwoCsvPromise;
}
async function getLocalizedData() {
  if (!localizedDataPromise) localizedDataPromise = fetch(chrome.runtime.getURL('pokemon_multilingual_names.json')).then(r => {
    if (!r.ok) throw new Error('Could not load multilingual Pokémon names data');
    return r.json();
  });
  return localizedDataPromise;
}
async function getUserAltNamesData() {
  if (!userAltNamesPromise) userAltNamesPromise = fetch(chrome.runtime.getURL('pokemon_user_alt_names.json')).then(r => {
    if (!r.ok) throw new Error('Could not load user alternate Pokémon names');
    return r.json();
  });
  return userAltNamesPromise;
}
async function getJapaneseNameList() {
  if (!japaneseDataPromise) {
    japaneseDataPromise = (async () => {
      const cached = await chrome.storage.local.get(['japaneseNameListCache', 'japaneseNameListCacheAt']);
      // Cache the source list locally so subsequent detections do not need another network request.
      if (Array.isArray(cached.japaneseNameListCache) && cached.japaneseNameListCache.length >= 900) {
        return cached.japaneseNameListCache;
      }
      const response = await fetch(JAPANESE_NAMES_URL, { cache: 'force-cache' });
      if (!response.ok) throw new Error(`Japanese names source returned HTTP ${response.status}`);
      const list = await response.json();
      if (!Array.isArray(list) || list.length < 900) throw new Error('Japanese names source returned an unexpected list');
      await chrome.storage.local.set({ japaneseNameListCache: list, japaneseNameListCacheAt: Date.now() });
      return list;
    })().catch(error => {
      japaneseDataPromise = null;
      throw error;
    });
  }
  return japaneseDataPromise;
}

function kanaToRomaji(value) {
  const text = String(value || '').replace(/[０-９]/g, ch => String.fromCharCode(ch.charCodeAt(0) - 0xFEE0));
  const kana = {
    'ア':'a','イ':'i','ウ':'u','エ':'e','オ':'o','カ':'ka','キ':'ki','ク':'ku','ケ':'ke','コ':'ko',
    'サ':'sa','シ':'shi','ス':'su','セ':'se','ソ':'so','タ':'ta','チ':'chi','ツ':'tsu','テ':'te','ト':'to',
    'ナ':'na','ニ':'ni','ヌ':'nu','ネ':'ne','ノ':'no','ハ':'ha','ヒ':'hi','フ':'fu','ヘ':'he','ホ':'ho',
    'マ':'ma','ミ':'mi','ム':'mu','メ':'me','モ':'mo','ヤ':'ya','ユ':'yu','ヨ':'yo','ラ':'ra','リ':'ri','ル':'ru','レ':'re','ロ':'ro',
    'ワ':'wa','ヲ':'o','ン':'n','ガ':'ga','ギ':'gi','グ':'gu','ゲ':'ge','ゴ':'go','ザ':'za','ジ':'ji','ズ':'zu','ゼ':'ze','ゾ':'zo',
    'ダ':'da','ヂ':'ji','ヅ':'zu','デ':'de','ド':'do','バ':'ba','ビ':'bi','ブ':'bu','ベ':'be','ボ':'bo','パ':'pa','ピ':'pi','プ':'pu','ペ':'pe','ポ':'po',
    'ァ':'a','ィ':'i','ゥ':'u','ェ':'e','ォ':'o','ャ':'ya','ュ':'yu','ョ':'yo','ッ':'','ー':'',
    'ヴ':'vu','ヵ':'ka','ヶ':'ke','・':'-',' ':'',
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9'
  };
  const digraphs = {
    'キャ':'kya','キュ':'kyu','キョ':'kyo','シャ':'sha','シュ':'shu','ショ':'sho','チャ':'cha','チュ':'chu','チョ':'cho',
    'ニャ':'nya','ニュ':'nyu','ニョ':'nyo','ヒャ':'hya','ヒュ':'hyu','ヒョ':'hyo','ミャ':'mya','ミュ':'myu','ミョ':'myo',
    'リャ':'rya','リュ':'ryu','リョ':'ryo','ギャ':'gya','ギュ':'gyu','ギョ':'gyo','ジャ':'ja','ジュ':'ju','ジョ':'jo',
    'ビャ':'bya','ビュ':'byu','ビョ':'byo','ピャ':'pya','ピュ':'pyu','ピョ':'pyo','ファ':'fa','フィ':'fi','フェ':'fe','フォ':'fo',
    'ウィ':'wi','ウェ':'we','ウォ':'wo','ヴァ':'va','ヴィ':'vi','ヴェ':'ve','ヴォ':'vo'
  };
  let out = '';
  const chars = Array.from(text);
  for (let i=0; i<chars.length; i++) {
    const ch = chars[i];
    const pair = ch + (chars[i+1] || '');
    if (digraphs[pair]) { out += digraphs[pair]; i++; continue; }
    if (ch === 'ッ') {
      const nextPair = pair;
      const next = digraphs[nextPair] || kana[chars[i+1]] || '';
      if (next) out += next[0];
      continue;
    }
    if (ch === 'ー') {
      const last = out.match(/[aeiou](?!.*[aeiou])/);
      if (last) out += last[0];
      continue;
    }
    if (ch === '♀' || ch === '♂' || ch === '：') continue;
    if (kana[ch] !== undefined) out += kana[ch];
    else if (/^[\u3040-\u30ff]$/.test(ch)) out += ch; // leave any uncommon kana visible rather than silently dropping it
    else out += ch;
  }
  return out.replace(/[^A-Za-z0-9-]/g, '').replace(/-+/g,'-').replace(/^-|-$/g,'').toLowerCase();
}

async function findLocalizedNames(referenceName, catchName) {
  try {
    const localData = await getLocalizedData();
    const localRecords = localData.records || [];
    const ref = normalizeSlug(referenceName);
    const catchKey = normalizeSlug(catchName);
    let localRecord = localRecords.find(r => normalizeSlug(r.slug) === ref);
    if (!localRecord && catchKey) localRecord = localRecords.find(r => Object.values(r.names || {}).some(n => normalizeSlug(n) === catchKey));
    if (!localRecord && ref) localRecord = localRecords.find(r => slugTokenKey(r.slug) === slugTokenKey(referenceName));

    // Prefer the Pokétwo-specific dataset: it contains aliases and form-specific slugs.
    let csvRecord = null;
    try {
      const csvRecords = await getPoketwoCsvRecords();
      csvRecord = csvRecords.find(r => normalizeSlug(r.slug) === ref);
      if (!csvRecord && ref) csvRecord = csvRecords.find(r => slugTokenKey(r.slug) === slugTokenKey(referenceName));
      if (!csvRecord && catchKey) csvRecord = csvRecords.find(r => ['name.en','name.en2','name.de','name.fr','name.ja_r','name.ja_t'].some(k => normalizeSlug(r[k]) === catchKey));
    } catch (csvError) {
      console.warn('[Pokémon Detector] Pokétwo names dataset unavailable; using bundled names:', csvError);
    }

    const rawNames = {};
    const localNames = localRecord?.names || {};
    // The custom option is independent from the built-in Alternate English source.
    // It uses the user's TXT aliases for exact base-species slugs, and the existing
    // Current Naming Rules alias for all other base species. For forms without a
    // separate entry, retain the already-resolved current catch name.
    try {
      const userAltData = await getUserAltNamesData();
      const altSlug = normalizeSlug(localRecord?.slug || ref);
      const altEntry = userAltData.by_slug?.[altSlug];
      rawNames.user_alt_names = cleanPokemonName(altEntry?.alt_name || catchName || '');
    } catch (altError) {
      console.warn('[Pokémon Detector] User alternate names unavailable; using current catch name:', altError);
      rawNames.user_alt_names = cleanPokemonName(catchName || '');
    }
    rawNames.english = localNames.english || '';
    rawNames.english_alternate = localNames.english_alternate || '';
    rawNames.german = localNames.german || '';
    rawNames.french = localNames.french || '';
    rawNames.japanese = localNames.japanese || '';
    rawNames.japanese_romanized = localNames.japanese_romanized || '';
    if (csvRecord) {
      // The reviewed bundled data is canonical and form-collapsed; online data only fills gaps.
      rawNames.english = rawNames.english || csvRecord['name.en'] || '';
      rawNames.english_alternate = rawNames.english_alternate || csvRecord['name.en2'] || '';
      rawNames.german = rawNames.german || csvRecord['name.de'] || '';
      rawNames.french = rawNames.french || csvRecord['name.fr'] || '';
      rawNames.japanese = rawNames.japanese || csvRecord['name.ja'] || '';
      rawNames.japanese_romanized = rawNames.japanese_romanized || csvRecord['name.ja_t'] || csvRecord['name.ja_r'] || '';
    }

    const labels = {
      english: 'English', english_alternate: 'Alternate English', user_alt_names: 'My Alt Names (TXT + Current Rules)', german: 'German', french: 'French',
      japanese_romanized: 'Japanese (Romanized)', japanese: 'Japanese'
    };
    const names = Object.entries(rawNames).filter(([language, name]) => labels[language] && String(name || '').trim()).map(([language, name]) => {
      const clean = cleanPokemonName(name);
      return { language, label: labels[language], name: clean };
    }).filter(item => item.name);

    // Fallback to the older Japanese species list only if the Pokétwo CSV did not provide Japanese.
    if (!names.some(item => item.language === 'japanese') && localRecord) {
      const englishBaseSlug = normalizeSlug(localRecord.names?.english || '');
      const recordSlug = normalizeSlug(localRecord.slug || '');
      const isBaseSpecies = !!englishBaseSlug && recordSlug === englishBaseSlug && !localRecord.is_form && !localRecord.mega && !localRecord.mega_x && !localRecord.mega_y;
      const dex = Number(localRecord.dex_number || localRecord.id);
      if (isBaseSpecies && Number.isInteger(dex) && dex > 0) {
        try {
          const japaneseList = await getJapaneseNameList();
          const japanese = String(japaneseList[dex - 1] || '').trim();
          if (japanese) {
            names.push({ language: 'japanese', label: 'Japanese (日本語)', name: japanese });
            const romanized = kanaToRomaji(japanese);
            if (romanized) names.push({ language: 'japanese_romanized', label: 'Japanese (Romanized)', name: stripLatinDiacritics(romanized) });
          }
        } catch (japaneseError) { console.warn('[Pokémon Detector] Japanese fallback unavailable:', japaneseError); }
      }
    }
    return names;
  } catch (e) {
    console.warn('[Pokémon Detector] Multilingual names could not be loaded:', e);
    return [];
  }
}

function pickLocalizedName(names, language, fallback) {
  const found = names.find(item => item.language === language && item.name);
  if (found) return cleanPokemonName(found.name).toLowerCase();
  const english = names.find(item => item.language === 'english' && item.name);
  return cleanPokemonName(english?.name || fallback || '').toLowerCase();
}
function formatOutput(catchName, bot, mode) {
  if (!catchName) return '';
  catchName = cleanPokemonName(catchName);
  if (!catchName) return '';
  if (mode === 'name') return catchName;
  return `${bot === 'bot2' ? 'p!c ' : '@Pokétwo#8236 c '}${catchName}`;
}
async function refreshLatestOutput() {
  if (!latest?.catchName && !latest?.pokemon) return;
  const prefs = await chrome.storage.local.get(['bot', 'outputMode', 'languageMode', 'selectedLanguage', 'blacklistPokemonNames']);
  const bot = prefs.bot || 'bot1';
  const mode = prefs.outputMode === 'name' ? 'name' : 'command';
  const langMode = ['current', 'selected', 'all', 'random'].includes(prefs.languageMode) ? prefs.languageMode : 'current';
  const lang = prefs.selectedLanguage || 'english';
  let outputName = cleanPokemonName(latest.catchName || latest.pokemon || '');
  if (langMode !== 'current') {
    const names = await findLocalizedNames(latest.pokemon, latest.catchName);
    latest = { ...latest, languageNames: names };
    if (langMode === 'selected') outputName = pickLocalizedName(names, lang, latest.catchName);
    else if (langMode === 'random' && names.length) outputName = cleanPokemonName(names[Math.floor(Math.random() * names.length)].name).toLowerCase();
    else if (langMode === 'all' && mode === 'name' && names.length) outputName = names.map(item => `${item.label}: ${cleanPokemonName(item.name)}`).join('\n');
  }
  const blacklistCandidates = [latest.pokemon, latest.catchName, outputName, ...(latest.languageNames || []).map(item => item.name)];
  const blacklisted = isPokemonBlacklisted(prefs.blacklistPokemonNames, blacklistCandidates);
  latest = { ...latest, blacklisted, displayName: cleanPokemonName(langMode === 'all' ? (latest.catchName || latest.pokemon || '') : outputName), command: blacklisted ? '.' : formatOutput(outputName, bot, mode) };
  await chrome.storage.local.set({ latest });
  const tabs = await getDiscordTabs();
  for (const tab of tabs) await sendTab(tab.id, { type: 'RESULT_UPDATED', latest });
}
async function sendTab(tabId, message) {
  try { await chrome.tabs.sendMessage(tabId, message); } catch (_) { /* tab may be loading or closed */ }
}
async function getDiscordTabs() {
  const tabs = await chrome.tabs.query({});
  return tabs.filter(tab => tab.id != null && tab.url && DISCORD_URL.test(tab.url));
}
async function syncMonitoring() {
  await loadSettings();
  const tabs = await getDiscordTabs();
  const selectedChannels = await getSelectedChannels();
  for (const tab of tabs) await sendTab(tab.id, { type: enabled && isSelectedChannel(tab.url, selectedChannels) ? 'START' : 'STOP' });
}
chrome.tabs.onActivated.addListener(() => { /* Do not stop monitoring when focus changes. */ });
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if ((changeInfo.status === 'complete' || changeInfo.url) && tab.url && DISCORD_URL.test(tab.url)) {
    await loadSettings();
    const selectedChannels = await getSelectedChannels();
    await sendTab(tabId, { type: enabled && isSelectedChannel(tab.url, selectedChannels) ? 'START' : 'STOP' });
  }
});
chrome.tabs.onRemoved.addListener(tabId => { /* Ignore queued jobs for tabs that are gone. */ });

async function handleSpawn(msg, sender, sendResponse) {
  await loadSettings();
  const selectedChannels = await getSelectedChannels();
  if (!enabled || !sender.tab?.id || !sender.tab.url || !DISCORD_URL.test(sender.tab.url) || !isSelectedChannel(sender.tab.url, selectedChannels)) {
    sendResponse({ ignored: true }); return;
  }
  const key = `${sender.tab.id}:${msg.messageId}:${msg.imageUrl}`;
  if (seen.has(key)) { sendResponse({ ignored: true }); return; }
  seen.add(key);
  const tabId = sender.tab.id;
  const job = async () => {
    // The tab might have been closed while waiting in the queue.
    const tab = await chrome.tabs.get(tabId).catch(() => null);
    if (!tab) return;
    latest = { status: 'processing', pokemon: '', catchName: '', score: null, command: '', error: '', imageUrl: msg.imageUrl, sourceTabId: tabId, sourceUrl: tab.url };
    await chrome.storage.local.set({ latest });
    await sendTab(tabId, { type: 'RESULT_UPDATED', latest });
    try {
      const imageResponse = await fetch(msg.imageUrl, { credentials: 'omit', cache: 'force-cache' });
      if (!imageResponse.ok) throw new Error(`Discord image fetch failed: HTTP ${imageResponse.status}`);
      const blob = await imageResponse.blob();
      if (!blob.type.startsWith('image/')) throw new Error(`URL did not return an image (${blob.type || 'unknown type'})`);
      const resultResponse = await fetch(`${SERVICE}/recognize`, { method: 'POST', headers: { 'Content-Type': blob.type || 'application/octet-stream' }, body: blob });
      const result = await resultResponse.json();
      if (!resultResponse.ok || !result.ok) throw new Error(result.error || `Local service HTTP ${resultResponse.status}`);
      await loadSettings();
      const best = result.results?.[0];
      const prefs = await chrome.storage.local.get(['bot', 'outputMode', 'languageMode', 'selectedLanguage', 'blacklistPokemonNames']);
      const bot = prefs.bot || 'bot1';
      const mode = prefs.outputMode === 'name' ? 'name' : 'command';
      const langMode = ['current', 'selected', 'all', 'random'].includes(prefs.languageMode) ? prefs.languageMode : 'current';
      const selectedLang = ['english', 'english_alternate', 'user_alt_names', 'german', 'french', 'japanese_romanized', 'japanese'].includes(prefs.selectedLanguage) ? prefs.selectedLanguage : 'english';
      const languageNames = best ? await findLocalizedNames(best.reference_name, best.catch_name) : [];
      let outputName = best ? cleanPokemonName(best.catch_name || best.reference_name || '') : '';
      if (best && langMode === 'selected') outputName = pickLocalizedName(languageNames, selectedLang, best.catch_name);
      else if (best && langMode === 'random' && languageNames.length) outputName = languageNames[Math.floor(Math.random() * languageNames.length)].name.toLowerCase();
      else if (best && langMode === 'all' && mode === 'name' && languageNames.length) outputName = languageNames.map(item => `${item.label}: ${item.name}`).join('\n');
      const blacklistCandidates = best ? [best.reference_name, best.catch_name, outputName, ...languageNames.map(item => item.name)] : [];
      const blacklisted = best ? isPokemonBlacklisted(prefs.blacklistPokemonNames, blacklistCandidates) : false;
      const command = best ? (blacklisted ? '.' : formatOutput(outputName, bot, mode)) : '';
      latest = { status: best ? 'recognized' : 'no-match', pokemon: best ? (languageNames.find(item => item.language === 'english')?.name || cleanPokemonName(best.reference_name || '')) : '', catchName: cleanPokemonName(best ? (best.catch_name || best.reference_name || '') : ''), displayName: cleanPokemonName(langMode === 'all' ? (best?.catch_name || best?.reference_name || '') : outputName), languageNames, languageMode: langMode, selectedLanguage: selectedLang, score: best?.score ?? null, command, blacklisted, error: '', imageUrl: msg.imageUrl, sourceTabId: tabId, sourceUrl: tab.url };
      await chrome.storage.local.set({ latest });
      await publishToDesktopPanel(latest, blob, tab);
      if (command && autoSwitch) {
        // Activate the source tab before copying so paste is immediately ready in that channel.
        await chrome.tabs.update(tabId, { active: true }).catch(() => {});
        if (tab.windowId != null) await chrome.windows.update(tab.windowId, { focused: true }).catch(() => {});
        await new Promise(resolve => setTimeout(resolve, 120));
      }
      if (command) await sendTab(tabId, { type: 'COPY_COMMAND', command });
      await sendTab(tabId, { type: 'RESULT_UPDATED', latest });
      // Also refresh any open Discord panel that is currently mounted.
      const discordTabs = await getDiscordTabs();
      for (const other of discordTabs) await sendTab(other.id, { type: 'RESULT_UPDATED', latest });
      return { ok: true, latest };
    } catch (error) {
      latest = { ...latest, status: 'error', error: String(error.message || error) };
      await chrome.storage.local.set({ latest });
      await sendTab(tabId, { type: 'RESULT_UPDATED', latest });
      return { ok: false, error: latest.error };
    }
  };
  processingQueue = processingQueue.then(job, job);
  const result = await processingQueue;
  sendResponse(result || { ignored: true });
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'SPAWN_IMAGE') { handleSpawn(msg, sender, sendResponse); return true; }
  (async () => {
    await loadSettings();
    if (msg.type === 'GET_STATE') {
      const tab = sender.tab || (await chrome.tabs.query({ active: true, lastFocusedWindow: true }))[0];
      const selectedChannels = await getSelectedChannels();
      const parsedChannel = channelFromUrl(tab?.url);
      const currentChannel = parsedChannel ? { ...parsedChannel, name: tab?.title || parsedChannel.channelId } : null;
      sendResponse({ enabled, autoSwitch, activeTabId: tab?.id ?? null, isDiscord: !!(tab?.url && DISCORD_URL.test(tab.url)), currentChannel, selectedChannels, outputMode, languageMode, selectedLanguage, latest });
    } else if (msg.type === 'TOGGLE_MONITORING') {
      enabled = !!msg.enabled;
      await chrome.storage.local.set({ monitoringEnabled: enabled });
      await syncMonitoring();
      sendResponse({ ok: true, enabled });
    } else if (msg.type === 'ADD_SELECTED_CHANNEL') {
      const channel = msg.channel && msg.channel.channelId && msg.channel.guildId ? msg.channel : null;
      const selectedChannels = await getSelectedChannels();
      if (channel && !selectedChannels.some(item => item.channelId === channel.channelId && item.guildId === channel.guildId)) selectedChannels.push(channel);
      await chrome.storage.local.set({ selectedChannels, selectedChannel: null });
      await syncMonitoring();
      sendResponse({ ok: true, selectedChannels });
    } else if (msg.type === 'REMOVE_SELECTED_CHANNEL') {
      const selectedChannels = (await getSelectedChannels()).filter(item => !(item.channelId === msg.channelId && item.guildId === msg.guildId));
      await chrome.storage.local.set({ selectedChannels, selectedChannel: null });
      await syncMonitoring();
      sendResponse({ ok: true, selectedChannels });
    } else if (msg.type === 'CLEAR_SELECTED_CHANNELS') {
      await chrome.storage.local.set({ selectedChannels: [], selectedChannel: null, monitoringEnabled: false });
      enabled = false;
      await syncMonitoring();
      sendResponse({ ok: true, selectedChannels: [] });
    } else if (msg.type === 'SET_AUTO_SWITCH') {
      autoSwitch = !!msg.enabled;
      await chrome.storage.local.set({ autoSwitch });
      sendResponse({ ok: true, autoSwitch });
    } else if (msg.type === 'SET_BOT') {
      await chrome.storage.local.set({ bot: msg.bot === 'bot2' ? 'bot2' : 'bot1' });
      await refreshLatestOutput();
      sendResponse({ ok: true });
    } else if (msg.type === 'SET_OUTPUT_MODE') {
      outputMode = msg.mode === 'name' ? 'name' : 'command';
      await chrome.storage.local.set({ outputMode });
      await refreshLatestOutput();
      sendResponse({ ok: true, outputMode });
    } else if (msg.type === 'SET_LANGUAGE_MODE') {
      languageMode = ['current', 'selected', 'all', 'random'].includes(msg.mode) ? msg.mode : 'current';
      await chrome.storage.local.set({ languageMode });
      await refreshLatestOutput();
      sendResponse({ ok: true, languageMode });
    } else if (msg.type === 'SET_LANGUAGE') {
      selectedLanguage = ['english', 'english_alternate', 'user_alt_names', 'german', 'french', 'japanese_romanized', 'japanese'].includes(msg.language) ? msg.language : 'english';
      await chrome.storage.local.set({ selectedLanguage });
      await refreshLatestOutput();
      sendResponse({ ok: true, selectedLanguage });
    } else if (msg.type === 'SAVE_POKEMON_BLACKLIST') {
      const blacklistPokemonNames = normalizeBlacklistNames(msg.names || []);
      await chrome.storage.local.set({ blacklistPokemonNames });
      await refreshLatestOutput();
      sendResponse({ ok: true, blacklistPokemonNames });
    } else if (msg.type === 'GET_LATEST') {
      sendResponse({ latest, enabled, autoSwitch });
    }
  })().catch(error => sendResponse({ ok: false, error: String(error) }));
  return true;
});
chrome.runtime.onInstalled.addListener(async () => {
  const state = await chrome.storage.local.get(['monitoringEnabled', 'autoSwitch', 'bot', 'latest', 'selectedChannel', 'outputMode', 'languageMode', 'selectedLanguage']);
  enabled = !!state.monitoringEnabled;
  autoSwitch = state.autoSwitch !== false;
  outputMode = state.outputMode === 'name' ? 'name' : 'command';
  languageMode = ['current', 'selected', 'all', 'random'].includes(state.languageMode) ? state.languageMode : 'current';
  selectedLanguage = ['english', 'english_alternate', 'user_alt_names', 'german', 'french', 'japanese_romanized', 'japanese'].includes(state.selectedLanguage) ? state.selectedLanguage : 'english';
  latest = state.latest || latest;
  const defaults = {};
  if (!state.bot) defaults.bot = 'bot1';
  if (state.autoSwitch === undefined) defaults.autoSwitch = true;
  if (state.outputMode === undefined) defaults.outputMode = 'command';
  if (state.languageMode === undefined) defaults.languageMode = 'current';
  if (state.selectedLanguage === undefined) defaults.selectedLanguage = 'english';
  if (Object.keys(defaults).length) await chrome.storage.local.set(defaults);
  await syncMonitoring();
});
