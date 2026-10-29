const SERVICE = 'http://127.0.0.1:8765';
let enabled = false;
let autoSwitch = true;
let outputMode = 'command'; // 'command' or 'name'
let latest = { status: 'idle', pokemon: '', score: null, command: '', error: '' };
const seen = new Set();
let processingQueue = Promise.resolve();
const DISCORD_URL = /^https:\/\/(?:[^/]+\.)?discord\.com\//i;

function channelFromUrl(url) {
  const match = String(url || '').match(/^https:\/\/(?:[^/]+\.)?discord\.com\/channels\/(\d+)\/(\d+)/i);
  return match ? { guildId: match[1], channelId: match[2], url: `https://discord.com/channels/${match[1]}/${match[2]}` } : null;
}
async function loadSettings() {
  const state = await chrome.storage.local.get(['monitoringEnabled', 'autoSwitch', 'latest', 'selectedChannel', 'outputMode']);
  enabled = !!state.monitoringEnabled;
  autoSwitch = state.autoSwitch !== false;
  outputMode = state.outputMode === 'name' ? 'name' : 'command';
  latest = state.latest || latest;
}
async function getSelectedChannel() {
  const state = await chrome.storage.local.get(['selectedChannel']);
  return state.selectedChannel || null;
}
function isSelectedChannel(url, selected) {
  const current = channelFromUrl(url);
  return !!(current && selected && current.channelId === selected.channelId && current.guildId === selected.guildId);
}
function formatOutput(catchName, bot, mode) {
  if (!catchName) return '';
  if (mode === 'name') return catchName;
  return `${bot === 'bot2' ? 'p!c ' : '@Pokétwo#8236 c '}${catchName}`;
}
async function refreshLatestOutput() {
  if (!latest?.catchName) return;
  const prefs = await chrome.storage.local.get(['bot', 'outputMode']);
  const bot = prefs.bot || 'bot1';
  const mode = prefs.outputMode === 'name' ? 'name' : 'command';
  latest = { ...latest, command: formatOutput(latest.catchName, bot, mode) };
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
  const selected = await getSelectedChannel();
  for (const tab of tabs) await sendTab(tab.id, { type: enabled && isSelectedChannel(tab.url, selected) ? 'START' : 'STOP' });
}
chrome.tabs.onActivated.addListener(() => { /* Do not stop monitoring when focus changes. */ });
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if ((changeInfo.status === 'complete' || changeInfo.url) && tab.url && DISCORD_URL.test(tab.url)) {
    await loadSettings();
    const selected = await getSelectedChannel();
    await sendTab(tabId, { type: enabled && isSelectedChannel(tab.url, selected) ? 'START' : 'STOP' });
  }
});
chrome.tabs.onRemoved.addListener(tabId => { /* Ignore queued jobs for tabs that are gone. */ });

async function handleSpawn(msg, sender, sendResponse) {
  await loadSettings();
  const selected = await getSelectedChannel();
  if (!enabled || !sender.tab?.id || !sender.tab.url || !DISCORD_URL.test(sender.tab.url) || !isSelectedChannel(sender.tab.url, selected)) {
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
      const prefs = await chrome.storage.local.get(['bot', 'outputMode']);
      const bot = prefs.bot || 'bot1';
      const mode = prefs.outputMode === 'name' ? 'name' : 'command';
      const command = best ? formatOutput(best.catch_name, bot, mode) : '';
      latest = { status: best ? 'recognized' : 'no-match', pokemon: best?.reference_name || '', catchName: best?.catch_name || '', score: best?.score ?? null, command, error: '', imageUrl: msg.imageUrl, sourceTabId: tabId, sourceUrl: tab.url };
      await chrome.storage.local.set({ latest });
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
      const selectedChannel = await getSelectedChannel();
      sendResponse({ enabled, autoSwitch, activeTabId: tab?.id ?? null, isDiscord: !!(tab?.url && DISCORD_URL.test(tab.url)), currentChannel: channelFromUrl(tab?.url), selectedChannel, outputMode, latest });
    } else if (msg.type === 'TOGGLE_MONITORING') {
      enabled = !!msg.enabled;
      await chrome.storage.local.set({ monitoringEnabled: enabled });
      await syncMonitoring();
      sendResponse({ ok: true, enabled });
    } else if (msg.type === 'SET_SELECTED_CHANNEL') {
      const selectedChannel = msg.channel && msg.channel.channelId && msg.channel.guildId ? msg.channel : null;
      await chrome.storage.local.set({ selectedChannel });
      await syncMonitoring();
      sendResponse({ ok: true, selectedChannel });
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
    } else if (msg.type === 'GET_LATEST') {
      sendResponse({ latest, enabled, autoSwitch });
    }
  })().catch(error => sendResponse({ ok: false, error: String(error) }));
  return true;
});
chrome.runtime.onInstalled.addListener(async () => {
  const state = await chrome.storage.local.get(['monitoringEnabled', 'autoSwitch', 'bot', 'latest', 'selectedChannel', 'outputMode']);
  enabled = !!state.monitoringEnabled;
  autoSwitch = state.autoSwitch !== false;
  outputMode = state.outputMode === 'name' ? 'name' : 'command';
  latest = state.latest || latest;
  const defaults = {};
  if (!state.bot) defaults.bot = 'bot1';
  if (state.autoSwitch === undefined) defaults.autoSwitch = true;
  if (state.outputMode === undefined) defaults.outputMode = 'command';
  if (Object.keys(defaults).length) await chrome.storage.local.set(defaults);
  await syncMonitoring();
});
