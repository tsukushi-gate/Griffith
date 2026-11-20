---
title: 'HDL Equivalence Verification'
description: 'Installing iverilog / ghdl and running three-way equivalence has been merged into "HDL Export and Equivalence Verification".'
---

:::note[This page has moved]
**→ [HDL Export and Equivalence Verification](/en/guides/hdl-export/)**

Installing the simulators (iverilog / ghdl), generating Verilog / VHDL, running
**three-way equivalence (PyTorch hard eval ⇔ `ir.simulate` ⇔ HDL simulation)**,
and the common pitfalls are all consolidated on that page.
To avoid documenting the same procedure in two places, this page keeps only the entry point.
:::

## Related pages

| Goal | Page |
|---|---|
| Walk through it hands-on once | [Tutorial 3: Exporting to HDL](/en/tutorials/hdl-export/) |
| Look up the IR spec behind the generated circuit | [IR schema](/en/reference/ir-schema/) |
| Burn the generated RTL onto real hardware | [Burning to an FPGA](/en/guides/fpga/) |
| Learn how to write equivalence checks as tests | [Test structure (L1–L5)](/en/contributing/testing/) |

<!-- i18n:source=guides/hdl.md hash=ebded24aa167 -->
