# F-Droid metadata

`io.github.derweh.bayesianbahn.yml` is the source of truth for our entry in
[fdroiddata](https://gitlab.com/fdroid/fdroiddata). The copy that lives in the
fdroiddata fork is a mirror — edit it here, then sync.

## Before pushing anything

```sh
pixi run fdroid-check          # or: tools/fdroid-check.sh
tools/fdroid-check.sh --fix    # apply the canonical formatting
```

This runs the same `fdroid rewritemeta` and `fdroid lint` that fdroiddata's
merge-request pipeline runs, using fdroidserver *master* against fdroiddata's
own category/anti-feature registries, so a green run here means a green
pipeline there. The same script runs in CI (`.github/workflows/fdroid-metadata.yml`),
including a weekly scheduled run — fdroiddata tracks fdroidserver master, so
this file can stop being canonical without anyone touching it.

## Two ways this file breaks silently

**Line endings.** fdroiddata's pipeline byte-compares the file and greps it with
anchored patterns. One CR makes the greps miss and produces a `rewritemeta`
failure whose diff looks empty. `.gitattributes` pins `fdroid/*.yml` to LF, and
the check script rejects CRs outright.

**Trailing whitespace is significant.** When a line exceeds the YAML emitter's
width, `rewritemeta` folds it and leaves a trailing space after the key:

```yaml
Binaries: ⟵ this space is required
  https://github.com/DerWeh/BayesianBahn/releases/download/v%v/BayesianBahn-v%v.apk
```

Do not point a trailing-whitespace stripper (editor-on-save, a
`trailing-whitespace` pre-commit hook) at this file. Whether a line gets folded
depends on the *ruamel.yaml version*, not on fdroidserver — Debian trixie's
0.18.x folds where 0.17.x does not, which is why `tools/fdroid-check.sh` pins it.

## Syncing to the fdroiddata fork

Copy through `curl`/`tr` rather than a file manager or editor, so neither CRs
nor a stripped trailing space sneak in:

```sh
cd <fdroiddata fork>
git checkout io.github.derweh.bayesianbahn
curl -sL https://raw.githubusercontent.com/DerWeh/BayesianBahn/HEAD/fdroid/io.github.derweh.bayesianbahn.yml \
  | tr -d '\r' > metadata/io.github.derweh.bayesianbahn.yml
grep -c $'\r' metadata/io.github.derweh.bayesianbahn.yml   # must print 0
git commit -am "BayesianBahn: update metadata" && git push
```

## Signing key

`AllowedAPKSigningKeys` pins the app to one certificate permanently: F-Droid
will refuse any future APK signed with a different key, and users would have to
uninstall and reinstall. Keep the keystore **and its passwords** backed up off
this machine. The release workflow verifies every published APK against the
value in this file before creating the GitHub release.
