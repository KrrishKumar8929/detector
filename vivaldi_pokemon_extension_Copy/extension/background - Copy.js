const SERVICE = 'http://127.0.0.1:8765';
let enabled = false;
let activeTabId = null;
let latest = { status: 'idle', pokemon: '', score: null, command: '', error: '' };
const seen = new Set();

async function getActiveDiscordTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab && tab.url && /^https:\/\/(?:[^/]+\.)?discord\.com\//i.test(tab.url) ? tab : null;
}
async function sendTab(tabId, message) {
  try { await chrome.tabs.sendMessage(tabId, message); } catch (_) { /* tab may not have content script yet */ }
}
async function refreshActiveTab() {
  const tab = await getActiveDiscordTab();
  const nextId = tab ? tab.id : null;
  if (activeTabId !== null && activeTabId !== nextId) await sendTab(activeTabId, { type: 'STOP' });
  activeTabId = nextId;
  if (enabled && activeTabId !== null) await sendTab(activeTabId, { type: 'START' });
  await chrome.storage.local.set({ monitoringEnabled: enabled, latest });
}
chrome.tabs.onActivated.addListener(() => { refreshActiveTab(); });
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'complete' && tabId === activeTabId) refreshActiveTab();
});
chrome.windows.onFocusChanged.addListener(() => refreshActiveTab());
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    if (msg.type === 'GET_STATE') {
      const tab = await getActiveDiscordTab();
      sendResponse({ enabled, activeTabId: tab?.id ?? null, isDiscord: !!tab, latest });
    } else if (msg.type === 'TOGGLE_MONITORING') {
      enabled = !!msg.enabled;
      if (!enabled && activeTabId !== null) await sendTab(activeTabId, { type: 'STOP' });
      await refreshActiveTab();
      sendResponse({ ok: true, enabled });
    } else if (msg.type === 'SET_BOT') {
      await chrome.storage.local.set({ bot: msg.bot === 'bot2' ? 'bot2' : 'bot1' });
      sendResponse({ ok: true });
    } else if (msg.type === 'SPAWN_IMAGE') {
      if (!enabled || sender.tab?.id !== activeTabId) return sendResponse({ ignored: true });
      const key = `${sender.tab.id}:${msg.messageId}:${msg.imageUrl}`;
      if (seen.has(key)) return sendResponse({ ignored: true });
      seen.add(key);
      try {
        const imageResponse = await fetch(msg.imageUrl, { credentials: 'omit', cache: 'force-cache' });
        if (!imageResponse.ok) throw new Error(`Discord image fetch failed: HTTP ${imageResponse.status}`);
        const blob = await imageResponse.blob();
        if (!blob.type.startsWith('image/')) throw new Error(`URL did not return an image (${blob.type || 'unknown type'})`);
        const resultResponse = await fetch(`${SERVICE}/recognize`, { method: 'POST', headers: { 'Content-Type': blob.type || 'application/octet-stream' }, body: blob });
        const result = await resultResponse.json();
        if (!resultResponse.ok || !result.ok) throw new Error(result.error || `Local service HTTP ${resultResponse.status}`);
        if (!enabled || sender.tab?.id !== activeTabId) return sendResponse({ ignored: true });
        const best = result.results?.[0];
        const prefs = await chrome.storage.local.get(['bot']);
        const bot = prefs.bot || 'bot1';
        const command = best ? `${bot === 'bot2' ? 'p!c ' : '@Pokétwo#8236 c '}${best.catch_name}` : '';
        latest = { status: best ? 'recognized' : 'no-match', pokemon: best?.reference_name || '', catchName: best?.catch_name || '', score: best?.score ?? null, command, error: '' };
        await chrome.storage.local.set({ latest });
        chrome.runtime.sendMessage({ type: 'RESULT_UPDATED', latest }).catch(() => {});
        sendResponse({ ok: true, latest });
      } catch (error) {
        latest = { ...latest, status: 'error', error: String(error.message || error) };
        await chrome.storage.local.set({ latest });
        chrome.runtime.sendMessage({ type: 'RESULT_UPDATED', latest }).catch(() => {});
        sendResponse({ ok: false, error: latest.error });
      }
    }
  })();
  return true;
});
chrome.runtime.onInstalled.addListener(async () => {
  const state = await chrome.storage.local.get(['monitoringEnabled', 'bot', 'latest']);
  enabled = !!state.monitoringEnabled;
  latest = state.latest || latest;
  if (!state.bot) await chrome.storage.local.set({ bot: 'bot1' });
  refreshActiveTab();
});
