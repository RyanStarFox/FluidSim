# Algorithm Source Copies

This directory preserves the branch-specific `framework.py` implementations used
for the final FLIP/APIC/PolyPIC comparison.

## Files

- `flip/framework.py`: FLIP baseline exported from branch `main`.
- `apic/framework.py`: APIC implementation exported from branch `apic-branch`.
- `polypic/framework.py`: PolyPIC implementation exported from branch `polypic`.

The repository root `framework.py` belongs to the currently checked-out final
report branch and should not be treated as the only algorithm implementation.
For grading or reproduction, use the algorithm-specific copies in this
directory.

## Example Commands

```bash
python submission/code/flip/framework.py all 0.97
python submission/code/apic/framework.py all 0.97
python submission/code/polypic/framework.py all 0.97 polypic
```

The committed outputs in `output/*/ratio_970/` were generated from these
algorithm branches and are summarized in the final PDF report.
