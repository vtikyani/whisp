import { render } from "preact";
import { html } from "./html.js";
import { App } from "./app.js";
import { connectBridge } from "./bridge.js";
import { boot } from "./store.js";

const root = document.getElementById("app");

try {
  await connectBridge();
  await boot();
  render(html`<${App} />`, root);
} catch (error) {
  console.error(`Whisp's window couldn't start: ${error && error.message}`);
  render(
    html`<div class="boot-error">
      <div>
        <h1 class="page-title">Whisp's window couldn't start</h1>
        <p class="page-desc">${(error && error.message) || "Something went wrong."} Close the window and open it again from the tray.</p>
      </div>
    </div>`,
    root,
  );
}
