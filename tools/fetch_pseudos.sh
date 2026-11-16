#!/bin/sh
# Downloads the PAW datasets used by the tests and examples into pseudos/ (or the directory given as argument).
#
# The datasets are NOT part of this repository and are NOT covered by its license. They belong to the JTH table
# v1.1 (F. Jollet, M. Torrent, N. Holzwarth, Comput. Phys. Commun. 185, 1246 (2014)) and are fetched from the
# PseudoDojo repositories at fixed revisions; the checksums guarantee that the tests see the files they were
# written for.
set -eu

DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/pseudos}"
LDA="https://raw.githubusercontent.com/PseudoDojo/ATOMPAW-LDA-JTHv1.1/cb2649791046ca6e45dcdd262fe0974346aabfab"
PBE="https://raw.githubusercontent.com/PseudoDojo/ATOMPAW-PBE-JTHv1.1/a8a1d2d8d6ac746d28fdcb517e9251a7da9ffea8"

FILES="
$LDA/Al/Al.LDA_PW-JTH.xml 829f2e3661416425fd8deabf27dc02223bd3dfa44cbe3cfb102361e8ecd79a45
$LDA/Fe/Fe.LDA_PW-JTH.xml 5001967a36af97a2763c69260674f54b8f62bc6b5f2a3b36c5f985e779c22fd4
$LDA/Ni/Ni.LDA_PW-JTH.xml 1609a03134a8a38990237b191abb0c08f5a8ebb0604c30d1a4ef926e4a64466c
$LDA/H/H.LDA_PW-JTH.xml 26ef5b6c4ef2f4ae26cd123f7c3d04fd21bd21a8905bf6b1bbce5c84253bfebe
$PBE/Al/Al.GGA_PBE-JTH.xml cc5334b78985b50c2517f87c302762c1e8b10c0f09956eaf446aa32cefa9d28c
$PBE/Fe/Fe.GGA_PBE-JTH.xml d80d912c6490e3074c210bd7bdafdcc61f2246300222437e6be7f61bcd49d35d
$PBE/Ni/Ni.GGA_PBE-JTH.xml 6df23ac8120a70d6fdf1d15c7707e8ff32e2f940d4c1e374d795e34e52382a5f
"

sha256() {
  if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
download() {
  if command -v curl > /dev/null 2>&1; then curl -fsSL -o "$2" "$1"; else wget -q -O "$2" "$1"; fi
}

mkdir -p "$DIR"
echo "$FILES" | while read -r url sum; do
  [ -n "$url" ] || continue
  name="$(basename "$url")"
  target="$DIR/$name"
  if [ -f "$target" ] && [ "$(sha256 "$target")" = "$sum" ]; then
    echo "ok       $name"
    continue
  fi
  download "$url" "$target.part"
  if [ "$(sha256 "$target.part")" != "$sum" ]; then
    rm -f "$target.part"
    echo "ERROR: checksum mismatch for $name" >&2
    exit 1
  fi
  mv "$target.part" "$target"
  echo "fetched  $name"
done
echo "PAW datasets are in $DIR"
