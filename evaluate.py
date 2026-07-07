import argparse
import glob
import numpy as np
import os
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from typing import Any


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default=None, required=True)
    parser.add_argument("--pred_root_path", type=str, default=None, required=True)

    return parser.parse_args()


def normalize_nodes(
    nodes,
    flip_y=True,
    bbox=None
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x, y = nodes[:, 0], nodes[:, 1]
    if bbox is None:
        minx, maxx = min(x), max(x)
        miny, maxy = min(y), max(y)
    else:
        minx, maxx = bbox[0], bbox[1]
        miny, maxy = bbox[2], bbox[3]

    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)

    return np.stack([x, y], axis=1), (minx, maxx, miny, maxy)


def parse_molblock(molblock: str) -> dict[tuple[int, int], Any]:
    lines = molblock.split('\n')
    stereo_bonds = {}

    for i, line in enumerate(lines):
        if line.endswith("V2000"):
            tokens = line.split()
            num_atoms = int(tokens[0])
            num_bonds = int(tokens[1])
            for bond_line in lines[i + 1 + num_atoms:i + 1 + num_atoms + num_bonds]:
                # bond_tokens = bond_line.strip().split()
                bond_tokens = [bond_line[:3], bond_line[3:6], bond_line[6:9], bond_line[9:12]]
                start, end, bond_type, stereo = [int(token) for token in bond_tokens]

                if bond_type == 1:
                    if stereo == 0:
                        continue

                    if stereo == 1:
                        etype = 5
                    elif stereo == 6:
                        etype = 6
                    elif stereo == 4:
                        etype = 8
                    else:
                        raise ValueError(f"Unsupported stereo type: {stereo}")
                    stereo_bonds[(start - 1, end - 1)] = etype
            break
    return stereo_bonds

def _get_norm_coords(mol) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    conf= mol.GetConformer()
    coords = []
    for i, a in enumerate(mol.GetAtoms()):
        coord = conf.GetAtomPosition(i)
        coords.append([coord.x, coord.y])
    coords = np.array(coords, dtype=np.float32)
    coords, bbox = normalize_nodes(coords)

    return coords, bbox


def _atom_equal(a_pred, a_gt) -> bool:
    # symbol_pred = a_pred.GetPropsAsDict().get("molFileAlias", a_pred.GetSymbol())
    # symbol_gt = a_gt.GetPropsAsDict().get("molFileAlias", a_gt.GetSymbol())
    symbol_pred = "R" if "molFileAlias" in a_pred.GetPropsAsDict() else a_pred.GetSymbol()
    symbol_gt = "R" if "molFileAlias" in a_gt.GetPropsAsDict() else a_gt.GetSymbol()

    if not symbol_pred.lower() == symbol_gt.lower():
        return False
    if not a_pred.GetFormalCharge() == a_gt.GetFormalCharge():
        return False
    if not a_pred.GetNumRadicalElectrons() == a_gt.GetNumRadicalElectrons():
        return False

    return True


def _get_bond_type(b, stereo_bond_override: int) -> float:
    if not b:
        return 0.0

    bond_type = b.GetBondTypeAsDouble()
    if bond_type == 1.5:
        bond_type = 4

    if bond_type == 2:
        if b.GetStereo() == Chem.BondStereo.STEREOANY:
            bond_type = 7

    assert stereo_bond_override in [0, 5, 6, 8]
    if stereo_bond_override:
        bond_type = stereo_bond_override

    return bond_type


def compare_molblocks(molblock_pred: str, molblock_gt: str) -> dict[str, Any]:
    # TODO: check how Hs are exactly handled
    mol_pred = Chem.MolFromMolBlock(molblock_pred, sanitize=False, removeHs=False, strictParsing=True)
    mol_gt = Chem.MolFromMolBlock(molblock_gt, sanitize=False, removeHs=False, strictParsing=True)

    stereo_bonds_pred = parse_molblock(molblock_pred)
    stereo_bonds_gt = parse_molblock(molblock_gt)

    if mol_pred is None:
        metrics = {
            "atom_precision": 0.0,
            "atom_recall": 0.0,
            "atom_f1": 0.0,
            "bond_precision": 0.0,
            "bond_recall": 0.0,
            "bond_f1": 0.0,
            "exact_match": 0.0
        }
        return metrics

    n_atom_pred = mol_pred.GetNumAtoms()
    n_atom_gt = mol_gt.GetNumAtoms()
    assert n_atom_pred == len(mol_pred.GetAtoms())
    assert n_atom_gt == len(mol_gt.GetAtoms())

    coords_pred, bbox_pred = _get_norm_coords(mol_pred)
    coords_gt, bbox_gt = _get_norm_coords(mol_gt)

    atom_costs = np.ones((n_atom_pred, n_atom_gt), dtype=np.float32) * 1e3
    for i, coord_pred in enumerate(coords_pred):
        for j, coord_gt in enumerate(coords_gt):
            atom_costs[i, j] = np.linalg.norm(coord_gt - coord_pred)

    row_ind, col_ind = linear_sum_assignment(atom_costs)
    # [print(f"{r}, {c}") for r, c in zip(row_ind, col_ind)]

    atom_precisions = np.zeros(n_atom_pred, dtype=np.float32)
    atom_recalls = np.zeros(n_atom_gt, dtype=np.float32)
    forward_map = {}
    reverse_map = {}
    for r, c in zip(row_ind, col_ind):
        forward_map[r] = c
        reverse_map[c] = r
        a_pred = mol_pred.GetAtomWithIdx(int(r))
        a_gt = mol_gt.GetAtomWithIdx(int(c))

        if _atom_equal(a_pred, a_gt):
            atom_precisions[r] = 1.0
            atom_recalls[c] = 1.0

    atom_precision = np.mean(atom_precisions) if atom_precisions.size else 0.0
    atom_recall = np.mean(atom_recalls) if atom_recalls.size else 0.0
    if atom_precision == 0.0 and atom_recall == 0.0:
        atom_f1 = 0.0
    else:
        atom_f1 = 2 * atom_precision * atom_recall / (atom_precision + atom_recall)

    # e.g., predicted bond (1 , 2) <=> gt bond (3, 4)
    bond_precisions = []
    bond_recalls = []
    for b_pred in mol_pred.GetBonds():
        begin_atom_i_pred = b_pred.GetBeginAtomIdx()
        end_atom_i_pred = b_pred.GetEndAtomIdx()
        try:
            begin_atom_i_gt = int(forward_map[begin_atom_i_pred])
            end_atom_i_gt = int(forward_map[end_atom_i_pred])
        except KeyError:
            bond_precisions.append(0.0)
            continue

        b_gt = mol_gt.GetBondBetweenAtoms(
            begin_atom_i_gt,
            end_atom_i_gt
        )
        stereo_bond_type_pred = stereo_bonds_pred.get((begin_atom_i_pred, end_atom_i_pred), 0)
        stereo_bond_type_gt = stereo_bonds_gt.get((begin_atom_i_gt, end_atom_i_gt), 0)
        b_type_pred = _get_bond_type(b_pred, stereo_bond_type_pred)
        b_type_gt = _get_bond_type(b_gt, stereo_bond_type_gt)

        if b_gt and b_type_pred == b_type_gt:
            bond_precisions.append(1.0)
        else:
            bond_precisions.append(0.0)

    for b_gt in mol_gt.GetBonds():
        begin_atom_i_gt = b_gt.GetBeginAtomIdx()
        end_atom_i_gt = b_gt.GetEndAtomIdx()
        try:
            begin_atom_i_pred = int(reverse_map[begin_atom_i_gt])
            end_atom_i_pred = int(reverse_map[end_atom_i_gt])
        except KeyError:
            bond_recalls.append(0.0)
            continue

        b_pred = mol_pred.GetBondBetweenAtoms(
            begin_atom_i_pred,
            end_atom_i_pred
        )
        stereo_bond_type_pred = stereo_bonds_pred.get((begin_atom_i_pred, end_atom_i_pred), 0)
        stereo_bond_type_gt = stereo_bonds_gt.get((begin_atom_i_gt, end_atom_i_gt), 0)
        b_type_pred = _get_bond_type(b_pred, stereo_bond_type_pred)
        b_type_gt = _get_bond_type(b_gt, stereo_bond_type_gt)

        if b_pred and b_type_pred == b_type_gt:
            bond_recalls.append(1.0)
        else:
            bond_recalls.append(0.0)

    bond_precision = np.mean(bond_precisions) if bond_precisions else 0.0
    bond_recall = np.mean(bond_recalls) if bond_recalls else 0.0
    if bond_precision == 0.0 and bond_recall == 0.0:
        bond_f1 = 0.0
    else:
        bond_f1 = 2 * bond_precision * bond_recall / (bond_precision + bond_recall)

    exact_match = (atom_f1 == 1.0) and (bond_f1 == 1.0)
    exact_match = (atom_f1 == 1.0)

    metrics = {
        "atom_precision": atom_precision,
        "atom_recall": atom_recall,
        "atom_f1": atom_f1,
        "bond_precision": bond_precision,
        "bond_recall": bond_recall,
        "bond_f1": bond_f1,
        "exact_match": exact_match
    }

    return metrics


def main(args):
    test_path = args.test_path
    pred_root_path = args.pred_root_path

    exact_matches = {}
    atom_precisions= {}
    atom_recalls = {}
    atom_f1s = {}
    bond_precisions= {}
    bond_recalls = {}
    bond_f1s = {}

    test_filelist = sorted(glob.glob(os.path.join(test_path, "*.corrected.mol")))
    if True:
        for molfile_gt in test_filelist:
            # molfile_gt = line.strip().replace(".png", ".corrected.mol")
            molfile_pred = molfile_gt.replace(".corrected.mol", ".predicted.mol")
            molfile_pred = "/".join(molfile_pred.split("/")[1:])
            molfile_pred = os.path.join(pred_root_path, molfile_pred)

            with open(molfile_gt, "r") as f_gt:
                molblock_gt = f_gt.read()
            with open(molfile_pred, "r") as f_pred:
                molblock_pred = f_pred.read()
            metrics = compare_molblocks(molblock_pred, molblock_gt)
            mol_gt = Chem.MolFromMolBlock(molblock_gt, sanitize=False, removeHs=False, strictParsing=True)
            atom_count = mol_gt.GetNumAtoms()

            count = atom_count // 10 * 10
            count = min(count, 50)
            # count = 1
            count = os.path.basename(molfile_pred).split(".")[0]
            count = "-".join(count.split("-")[:-2])

            if count in exact_matches:
                exact_matches[count].append(metrics["exact_match"])
                atom_precisions[count].append(metrics["atom_precision"])
                atom_recalls[count].append(metrics["atom_recall"])
                atom_f1s[count].append(metrics["atom_f1"])
                bond_precisions[count].append(metrics["bond_precision"])
                bond_recalls[count].append(metrics["bond_recall"])
                bond_f1s[count].append(metrics["bond_f1"])
            else:
                exact_matches[count] = [metrics["exact_match"]]
                atom_precisions[count] = [metrics["atom_precision"]]
                atom_recalls[count] = [metrics["atom_recall"]]
                atom_f1s[count] = [metrics["atom_f1"]]
                bond_precisions[count] = [metrics["bond_precision"]]
                bond_recalls[count] = [metrics["bond_recall"]]
                bond_f1s[count] = [metrics["bond_f1"]]


            print(f"molfile_gt: {molfile_gt}, metrics: {metrics}")

    print(pred_root_path)
    # for count in sorted(exact_matches.keys()):
    #     # print(f"count: {count} - {count+19}, "
    #     print(f"count: {count}, occurrences: {len(exact_matches[count])}, "
    #           f"Exact matches: {np.mean(exact_matches[count]): .2f}, "
    #           # f"AP: {np.mean(atom_precisions[count]): .4f}, "
    #           # f"AR: {np.mean(atom_recalls[count]): .4f}, "
    #           f"Atom F1: {np.mean(atom_f1s[count]): .4f}, "
    #           # f"BP: {np.mean(bond_precisions[count]): .4f}, "
    #           # f"BR: {np.mean(bond_recalls[count]): .4f}, "
    #           f"Bond F1: {np.mean(bond_f1s[count]): .4f}")
    print("\t".join(sorted(exact_matches.keys())))
    print("Exact matches:")
    for count in sorted(exact_matches.keys()):
        print(f"{np.mean(exact_matches[count]): .2f}", end="\t")
    print("\n")
    print("Atom F1:")
    for count in sorted(exact_matches.keys()):
        print(f"{np.mean(atom_f1s[count]): .4f}", end="\t")
    print("\n")
    print("Bond F1:")
    for count in sorted(exact_matches.keys()):
        print(f"{np.mean(bond_f1s[count]): .4f}", end="\t")
    print("\n")

    print(f"Cumulative. "
          f"Exact matches: {np.mean([item for sublist in exact_matches.values() for item in sublist]): .2f}, "
          f"Atom F1: {np.mean([item for sublist in atom_f1s.values() for item in sublist]): .4f}, "
          f"Bond F1: {np.mean([item for sublist in bond_f1s.values() for item in sublist]): .4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)
