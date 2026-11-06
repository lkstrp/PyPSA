<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

Names are unique across entity types (every component type except Bus, Carrier,
SubNetwork, LineType and TransformerType). Importers rename clashing names
automatically and report them in `Network.deduplicated_names`. Code that
builds networks with possibly clashing names can run under
`pypsa.network.names.unchecked_names(n)` and then call `deduplicate_names(n)`.

::: pypsa.network.names.deduplicate_names

::: pypsa.network.names.unchecked_names
