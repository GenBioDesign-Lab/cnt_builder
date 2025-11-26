# Build CNT

Build armchair carbon nanotube (CNT) structures with optional functionalization. Output format: MOL2.

## Requirements

- Python 3
- NumPy
- RDKit (optional, for enhanced bond perception)

## Usage

```bash
python buildcnt.py [options] output.mol2
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n`, `--index` | Nanotube index n | 8 |
| `-l`, `--length` | CNT backbone length (Å) | 20.0 |
| `-f`, `--functionalization` | Type: `none`, `oh`, `cooh`, `coo`, `amide` | none |
| `--amide-angle` | Tilt angle for amide bond axis (degrees) | 60.0 |
| `--no-stagger` | Disable staggered amide plane arrangement | - |
| `--use-rdkit` | Use RDKit for bond perception | - |

### Examples

```bash
python buildcnt.py -n 8 -l 20.0 output.mol2
python buildcnt.py -n 6 -l 15.0 -f amide --amide-angle 45.0 cnt_amide.mol2
python buildcnt.py -n 8 -l 20.0 -f oh cnt_oh.mol2
```

## License

Freeware. You can use, modify and redistribute the source.

