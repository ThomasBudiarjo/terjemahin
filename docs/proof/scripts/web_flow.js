// Playwright walk through the terjemahin web UI: upload, translate, inspect stages, edit, re-typeset,
// download, then a second job with other methods picked in the UI.
// usage: NODE_PATH=/opt/node-tools/node_modules node web_flow.js http://127.0.0.1:8765 PAGE OUTDIR
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const [base, page01, outDir] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const log = (...a) => console.log(new Date().toISOString(), ...a);

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
  const page = await ctx.newPage();
  const shot = async (name, full = false) => {
    await page.screenshot({ path: path.join(outDir, name), fullPage: full });
    log("shot", name);
  };
  page.on("console", (m) => { if (m.type() === "error") log("console error:", m.text()); });

  await page.goto(base + "/");
  await page.waitForSelector("#m-detector");
  await shot("01-empty.png");

  await page.setInputFiles("#file", page01);
  await page.waitForFunction(() => !document.getElementById("go").disabled);
  const t0 = Date.now();
  await page.click("#go");
  await page.waitForFunction(() => /Running/.test(document.getElementById("msg").textContent), null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  await shot("02-running.png");
  await page.waitForFunction(() => !document.getElementById("bubbles").hidden, null, { timeout: 600000 });
  log("job 1 done in", (Date.now() - t0) / 1000, "s; msg:", await page.textContent("#msg"), "timings:", await page.textContent("#timings"));
  await page.waitForTimeout(800);
  await shot("03-done.png");
  await shot("04-done-full.png", true);

  for (const v of ["detect", "mask", "inpainted"]) {
    await page.selectOption("#rightview", v);
    await page.waitForTimeout(800);
    await shot(`05-view-${v}.png`);
  }
  await page.selectOption("#rightview", "final");

  // hand edits: fix bubble 13's translation to the GT reference, untick the handwritten SFX (14)
  const n = await page.$$eval("#rows tr", (r) => r.length);
  log("bubble rows:", n);
  await page.fill("#t-12", "Fufu, terus apa lagi?");
  if (n >= 14) await page.uncheck("#k-13");
  const t1 = Date.now();
  await page.click("#retype");
  await page.waitForFunction(() => /Re-typeset in/.test(document.getElementById("msg").textContent), null, { timeout: 120000 });
  log("re-typeset in", (Date.now() - t1) / 1000, "s; msg:", await page.textContent("#msg"));
  await page.waitForTimeout(800);
  await page.locator("#rows tr").nth(12).scrollIntoViewIfNeeded();
  await shot("06-retypeset.png");

  const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#download")]);
  const dlPath = path.join(outDir, "downloaded-" + dl.suggestedFilename());
  await dl.saveAs(dlPath);
  log("downloaded", dlPath, fs.statSync(dlPath).size, "bytes");

  // second job: other methods picked from the UI dropdowns
  await page.selectOption("#m-detector", "yolo");
  await page.selectOption("#m-translator", "m2m100");
  await page.selectOption("#m-inpainter", "telea");
  const t2 = Date.now();
  await page.click("#go");
  await page.waitForFunction(() => /Running/.test(document.getElementById("msg").textContent), null, { timeout: 60000 });
  await page.waitForFunction(() => !/Running|Waiting/.test(document.getElementById("msg").textContent), null, { timeout: 600000 });
  await page.waitForTimeout(800);
  log("job 2 done in", (Date.now() - t2) / 1000, "s; msg:", await page.textContent("#msg"), "url:", page.url());
  await shot("07-other-methods.png");

  // reload: the job comes back from the URL hash
  await page.reload();
  await page.waitForFunction(() => !document.getElementById("bubbles").hidden, null, { timeout: 60000 });
  log("reloaded:", await page.textContent("#msg"));

  // phone width
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(800);
  await shot("08-mobile.png");
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
