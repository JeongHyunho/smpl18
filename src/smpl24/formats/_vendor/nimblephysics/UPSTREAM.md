# NimblePhysics B3D schema pin

| Item | Value |
|---|---|
| Repository | `keenon/nimblephysics` |
| Commit | `c405b056fc35068027e03e0c384e84e12870b475` |
| Source path | `dart/proto/SubjectOnDisk.proto` |
| Expected source bytes | `10,574` |
| Expected SHA-256 | `37C6C94D879E2F6C7269FA868560919BEFCA0721C6AE204934953B7E29CB54BF` |
| Generated Python bytes | `8,204` |
| Generated Python SHA-256 | `8A621F9106611855DEB7208B6A4C582C3E19A90A991F17FF839DAA0C392D33FD` |
| Generator | `libprotoc 3.21.9` |
| License | upstream MIT license |
| Use | read-only decoding of `.b3d` containers by `smpl24.formats.b3d` |

Both files are verbatim upstream copies and are pinned by the hashes above; neither is edited
here, including the comments the `.proto` carries about where its fields originate. The
vendored `.proto` is the authoritative field-number source. The generated bindings must be
regenerated from this exact file; no code may restate a protobuf field number by hand.

## Regeneration

From the package root (`packages/smpl24`, or the repository root once the package is its own
repository), with a `protoc` whose `--version` prints `libprotoc 3.21.9`:

```powershell
protoc.exe `
  --proto_path=src/smpl24/formats/_vendor/nimblephysics `
  --python_out=src/smpl24/formats/_vendor/nimblephysics `
  src/smpl24/formats/_vendor/nimblephysics/SubjectOnDisk.proto
```

Afterwards compare the SHA-256 of `SubjectOnDisk_pb2.py` with the table.

## Outer B3D framing evidence

| Item | Value |
|---|---|
| Upstream source | `dart/biomechanics/SubjectOnDisk.cpp` |
| Commit | `c405b056fc35068027e03e0c384e84e12870b475` |
| Observed bytes | `148,603` |
| SHA-256 | `EEDAC7833CD553E3E328463C187DB00ED3D13D9174DCCD019734D8BB0F451C8D` |
| Evidence use | signed little-endian header length, fixed sensor/pass frame widths, trial offset formula |

The outer file layout (an 8-byte little-endian header length, the header message, then a flat
run of fixed-width sensor and processing-pass records) is taken from the upstream C++ source
named above, not from the `.proto`, which describes only the messages. `smpl24.formats.b3d`
implements that layout and checks it against the file's own length before reading a frame.
