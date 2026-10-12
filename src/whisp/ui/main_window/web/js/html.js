// JSX-like templates without a build step: html`<div class=${x}>...</div>`.
import { h, Fragment } from "preact";
import htm from "htm";

export const html = htm.bind(h);
export { Fragment };
