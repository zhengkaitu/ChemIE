import argparse
import traceback as tb
import json
import numpy as np
import os
from datasets import load_from_disk
from rdkit import Chem, RDLogger
from SmilesPE.pretokenizer import atomwise_tokenizer
from typing import Any, Dict, List, Tuple

RDLogger.DisableLog('rdApp.*')


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--expt_id", type=str, default=None, required=True)

    return parser.parse_args()


def parse_molblock(molblock: str) -> List[Tuple[int, int, int]]:
    # RDKit does not track BEGINWEDGE/BEGINDASH by default
    # need to manually parse this info from molfile, like in MolScribe

    lines = molblock.split('\n')
    stereo_bonds = []

    for i, line in enumerate(lines):
        if "BEGIN" in line and "CTAB" in line:
            _, _, _, num_atoms, num_bonds, _, _, _ = lines[i+1].split()
            num_atoms = int(num_atoms)
            num_bonds = int(num_bonds)
            continue
        if "BEGIN" in line and "BOND" in line:
            for bond_line in lines[i+1:i+1+num_bonds]:
                bond_tokens = bond_line.split()
                bond_type = int(bond_tokens[3])
                start = int(bond_tokens[4])
                end = int(bond_tokens[5])

                try:
                    stereo = bond_tokens[6]
                except IndexError:
                    continue

                if stereo.startswith("ENDPTS"):
                    continue

                if bond_type == 1:
                    # Up (wedge)
                    if stereo == "CFG=1":
                        etype = 5
                    # Down (dash)
                    elif stereo == "CFG=3":
                        etype = 6
                    # Either (wavy)
                    elif stereo == "CFG=2":
                        etype = 8
                    else:
                        raise ValueError(f"Unsupported stereo type: {stereo}, {molblock}")
                    stereo_bonds.append((start - 1, end - 1, etype))

            break

    return stereo_bonds


def get_edges(mol, molblock: str, inverse_map: list) -> List[List]:
    stereo_bonds = parse_molblock(molblock=molblock)
    stereo_bonds = {
        (start, end): bond_type
        for start, end, bond_type in stereo_bonds
    }

    edges = []
    for bond_i, bond in enumerate(mol.GetBonds()):
        bond_type = bond.GetBondTypeAsDouble()
        try:
            assert bond_type.is_integer() or bond_type == 1.5
        except AssertionError:
            print(f"{bond_type}, {molblock}")
            break

        if bond_type == 1.5:
            bond_type = 4
        else:
            bond_type = int(bond_type)

        if bond_type == 2:
            if bond.GetStereo() == Chem.BondStereo.STEREOANY:
                bond_type = 7

        begin_atom_i_in_mol = bond.GetBeginAtomIdx()
        end_atom_i_in_mol = bond.GetEndAtomIdx()
        begin_atom_i = inverse_map[bond.GetBeginAtomIdx()]
        end_atom_i = inverse_map[bond.GetEndAtomIdx()]
        # print(f"begin i: {bond.GetBeginAtomIdx()} -> {begin_atom_i}, "
        #       f"symbol: {mol.GetAtomWithIdx(begin_atom_i_in_mol).GetSymbol()} "
        #       f"end i: {bond.GetEndAtomIdx()} -> {end_atom_i}, "
        #       f"symbol: {mol.GetAtomWithIdx(end_atom_i_in_mol).GetSymbol()}")

        # Override with type 5 and 6, if in stereo_bonds
        bond_type = stereo_bonds.get(
            (begin_atom_i_in_mol, end_atom_i_in_mol),
            bond_type
        )

        edge = [begin_atom_i, end_atom_i, bond_type]
        edges.append(edge)

        props = bond.GetPropsAsDict()
        if "_MolFileBondEndPts" in props:
            assert "_MolFileBondAttach" in props
            endpts = props["_MolFileBondEndPts"][1:-1].split()
            count = int(endpts[0])
            assert len(endpts) == count + 1

            if props["_MolFileBondAttach"] == "ANY":
                bond_type = 9
            elif props["_MolFileBondAttach"] == "ALL":
                bond_type = 10
            else:
                raise NotImplementedError(props["_MolFileBondAttach"])

            for endpt in endpts[1:]:
                endpt_i_in_mol = int(endpt) - 1
                endpt_i = inverse_map[endpt_i_in_mol]

                edge = [begin_atom_i, endpt_i, bond_type]
                edges.append(edge)

    return edges


def get_row(example: Dict[str, Any], idx: int) -> Dict[str, str]:
    example_id = example["id"]
    molblock = example["mol"]

    # Disable sanitization to suppress aromatic labeling by rdkit.
    # The data provider should ensure the correctness of molfiles.
    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False, strictParsing=False)
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    # Chem.Kekulize(mol)
    # raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)

    # for atom in mol.GetAtoms():
    #     if atom.GetAtomicNum() == 0:
    #         props = atom.GetPropsAsDict()
    #         print(f"Atom {atom.GetIdx()}: {props}")
    # exit(0)

    try:
        raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=False, canonical=False)
    except RuntimeError:
        print(f"Runtime error for row idx: {idx}")
        tb.print_exc()
        raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=False, canonical=False, isomericSmiles=False)

    reordered_atom_i = eval(mol.GetProp("_smilesAtomOutputOrder"))
    inverse_map = np.argsort(reordered_atom_i).tolist()
    # print(reordered_atom_i)
    # print(inverse_map)

    smi = raw_smi
    superatoms = {}
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        props = atom.GetPropsAsDict()
        alias = props.get("molFileAlias")
        if alias:
            superatoms[inverse_map[atom_idx]] = alias
        else:
            dummy_label = props.get("dummyLabel")
            if dummy_label:
                superatoms[inverse_map[atom_idx]] = dummy_label

    # copied over from molscribe
    if superatoms:
        tokens = atomwise_tokenizer(smi)
        atom_idx = 0
        for i, t in enumerate(tokens):
            if t.isalpha() or t[0] == '[' or t == '*':
                if atom_idx in superatoms:
                    symbol = superatoms[atom_idx]
                    tokens[i] = f"[{symbol}]"
                atom_idx += 1
        smi = "".join(tokens)

    conf = mol.GetConformer()
    node_coords = []
    for atom_i in reordered_atom_i:
        coord = conf.GetAtomPosition(atom_i)
        node_coords.append([coord.x, coord.y])

    edges = get_edges(mol=mol, molblock=molblock, inverse_map=inverse_map)

    bracket_tokens = []
    bracket_coords = []
    for i, sg in enumerate(Chem.GetMolSubstanceGroups(mol)):
        brackets = sg.GetBrackets()
        if len(brackets) > 2:
            print(f"{len(brackets)} brackets found for row idx: {idx}")

        properties = sg.GetPropsAsDict()
        SCN = properties.get("CONNECT", "")  # superscript, essentially
        SMT = properties.get("LABEL", "")  # subscript, essentially
        # print(brackets)
        # use for loop to cover images with >2 brackets
        for bracket in brackets[:-1]:
            bracket_tokens.append(["<bra>"])
            bracket_coords.append([bracket[0].x, bracket[0].y])
            bracket_tokens.append(["<ket>"])
            bracket_coords.append([bracket[1].x, bracket[1].y])

        # lastly, attaching CONNECT and LABEL with the last <ket>
        # just to keep a record and ensure length consistency.
        # These will be further processed downstream

        # bracket_tokens.append(["<bra>"] + [token for token in SCN])
        # bracket_coords.append([brackets[-1][0].x, brackets[-1][0].y])
        # bracket_tokens.append(["<ket>"] + [token for token in str(SMT)])
        # bracket_coords.append([brackets[-1][1].x, brackets[-1][1].y])

        bracket_tokens.append(["<bra>"])
        bracket_coords.append([brackets[-1][0].x, brackets[-1][0].y])
        bracket_tokens.append(
            ["<ket>"] +
            ["<scn>"] + [token for token in str(SCN)] +
            ["<smt>"] + [token for token in str(SMT)]
        )
        bracket_coords.append([brackets[-1][1].x, brackets[-1][1].y])

    row = {
        "idx": idx,
        "id": example_id,
        "raw_SMILES": raw_smi,
        "SMILES": smi,
        "node_coords": json.dumps(node_coords, separators=(",", ":")),
        "bracket_tokens": json.dumps(bracket_tokens, separators=(",", ":")),
        "bracket_coords": json.dumps(bracket_coords, separators=(",", ":")),
        "edges": json.dumps(edges, separators=(",", ":"))
    }

    return row


def dataset2csv(dataset, ofn: str) -> None:
    updated_dataset = dataset.map(
        get_row,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names
    )
    updated_dataset.to_csv(ofn)


def aggregate_into_csv(args) -> None:
    csv_output_path = "data/hf"
    os.makedirs(csv_output_path, exist_ok=True)
    dataset_dirs = [
        ("data/hf/markushgrapher-synthetic-training", "train", "synthetic-train.processed.csv"),
        ("data/hf/markushgrapher-synthetic-training", "test", "synthetic-val.processed.csv")
        # ("data/hf/markushgrapher-synthetic", "test", "synthetic-test.processed.csv")
    ]

    for dataset_dir, split, csv_fn in dataset_dirs:
        ofn = os.path.join(csv_output_path, csv_fn)

        dataset = load_from_disk(dataset_dir)
        dataset = dataset[split]

        dataset2csv(dataset=dataset, ofn=ofn)


def main(args):
    aggregate_into_csv(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
