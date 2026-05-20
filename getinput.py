import numpy as np
import nibabel as nib
import nibabel.freesurfer.io as fsio
from scipy.spatial import KDTree
import pymeshlab
import json
import os
from collections import OrderedDict
import h5py
from tqdm import tqdm


class MeshHierarchy:
    def __init__(self):
        self.levels = OrderedDict()
        self.mapping = {}
        self.metadata = {
            'template': 'tpl-onavg',
            'original_resolution': '~40km'
        }

    def add_level(self, level, verts, faces, func=None):
        self.levels[level] = {
            'vertices': verts,
            'faces': faces,
            'function': func if func is not None else None
        }

    def add_mapping(self, from_level, to_level, indices, weights):
        key = f"L{from_level}_to_L{to_level}"
        self.mapping[key] = {
            'indices': indices,
            'weights': weights
        }

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        with h5py.File(os.path.join(output_dir, 'data.h5'), 'w') as hf:
            for level, data in self.levels.items():
                group = hf.create_group(f'level_{level}')
                group.create_dataset('vertices', data=data['vertices'], compression='gzip')
                group.create_dataset('faces', data=data['faces'], compression='gzip')
                if data['function'] is not None:
                    group.create_dataset('function', data=data['function'], compression='gzip')

                obj_path = os.path.join(output_dir, f'level_{level}_mesh.obj')
                with open(obj_path, 'w') as f:
                    for v in data['vertices']:
                        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                    for face in data['faces']:
                        f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

            map_group = hf.create_group('mappings')
            for key, mapping in self.mapping.items():
                sub_group = map_group.create_group(key)
                sub_group.create_dataset('indices', data=mapping['indices'], compression='gzip')
                sub_group.create_dataset('weights', data=mapping['weights'], compression='gzip')

        self.metadata.update({
            'num_levels': len(self.levels),
            'level_resolutions': {level: self._calculate_resolution(level)
                                  for level in self.levels.keys()},
            'vertex_counts': {level: len(data['vertices'])
                              for level, data in self.levels.items()}
        })
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _calculate_resolution(self, level):
        verts = self.levels[level]['vertices']
        faces = self.levels[level]['faces']

        edges = np.vstack([
            np.linalg.norm(verts[faces[:, 1]] - verts[faces[:, 0]], axis=1),
            np.linalg.norm(verts[faces[:, 2]] - verts[faces[:, 1]], axis=1),
            np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 2]], axis=1)
        ])
        return float(np.mean(edges))


def load_tpl_onavg_data(lh_surf_path, rh_surf_path, lh_func_path=None, rh_func_path=None):
    print("Loading tpl-onavg data...")

    def load_surface(surf_path):
        if surf_path.endswith('.gii'):
            gii = nib.load(surf_path)
            return gii.darrays[0].data, gii.darrays[1].data
        else:
            return fsio.read_geometry(surf_path)

    lh_verts, lh_faces = load_surface(lh_surf_path)
    rh_verts, rh_faces = load_surface(rh_surf_path)

    verts = np.vstack([lh_verts, rh_verts])
    faces = np.vstack([lh_faces, rh_faces + len(lh_verts)])

    func = None
    if lh_func_path and rh_func_path:
        def load_func(func_path):
            if func_path.endswith('.mgh'):
                return nib.load(func_path).get_fdata().squeeze()
            else:
                return nib.load(func_path).get_fdata().reshape(-1)

        lh_func = load_func(lh_func_path)
        rh_func = load_func(rh_func_path)
        func = np.vstack([lh_func, rh_func])
        if func.ndim == 1:
            func = func[:, np.newaxis]

    print(f"-> Vertices: {verts.shape[0]}, Faces: {faces.shape[0]}, "
          f"Signal dimension: {func.shape if func is not None else 'None'}")
    return verts, faces, func


def create_hierarchical_mesh(verts, faces, func=None, reduction_ratios=None):
    if reduction_ratios is None:
        reduction_ratios = [0.8, 0.6, 0.4, 0.2, 0.1]

    hierarchy = MeshHierarchy()
    max_level = len(reduction_ratios)
    hierarchy.add_level(max_level, verts, faces, func)
    print(f"Added highest resolution level {max_level} (Vertices: {len(verts)})")

    for i in tqdm(range(len(reduction_ratios) - 1, -1, -1), desc="Creating levels"):
        current_level = i + 1
        current_data = hierarchy.levels[current_level]

        new_verts, new_faces = simplify_mesh(
            current_data['vertices'],
            current_data['faces'],
            reduction_ratio=reduction_ratios[i],
            current_level=current_level
        )

        if current_data['function'] is not None:
            new_func, mapping = aggregate_signals(
                current_data['vertices'],
                new_verts,
                current_data['function'],
                k=max(3, int(8 * (len(current_data['vertices']) / len(new_verts))))
            )
        else:
            new_func = None
            mapping = {'indices': None, 'weights': None}

        hierarchy.add_level(i, new_verts, new_faces, new_func)
        hierarchy.add_mapping(current_level, i, mapping['indices'], mapping['weights'])

    return hierarchy


def simplify_mesh(verts, faces, reduction_ratio, current_level, quality_thr=0.7):
    target_faces = int(len(faces) * reduction_ratio)
    print(f"\nSimplifying level {current_level} -> {current_level - 1}: "
          f"{len(faces)} -> {target_faces} faces")

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces))

    # Use only basic mesh simplification to avoid unstable advanced features
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualitythr=quality_thr,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True
    )

    # For very coarse levels, add an additional simplification step
    if reduction_ratio < 0.3:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces * 0.9),
            qualitythr=0.5,
            preservenormal=True,
            preservetopology=True
        )

    simplified = ms.current_mesh()
    return simplified.vertex_matrix(), simplified.face_matrix()


def aggregate_signals(fine_verts, coarse_verts, fine_func, k=5, chunk_size=10000):
    print(f"Aggregating signals: {len(fine_verts)} -> {len(coarse_verts)} vertices (k={k})")

    coarse_tree = KDTree(coarse_verts)
    n_fine = len(fine_verts)
    n_time = fine_func.shape[1] if fine_func.ndim > 1 else 1

    coarse_func = np.zeros((len(coarse_verts), n_time))
    fine_to_coarse = np.zeros((n_fine, k), dtype=int)
    weights = np.zeros((n_fine, k))

    for i in tqdm(range(0, n_fine, chunk_size), desc="Processing vertex chunks"):
        chunk_end = min(i + chunk_size, n_fine)
        chunk_verts = fine_verts[i:chunk_end]

        dists, indices = coarse_tree.query(chunk_verts, k=k)

        sigma = np.median(dists) * 0.5
        chunk_weights = np.exp(-(dists ** 2) / (2 * sigma ** 2 + 1e-10))
        chunk_weights = np.nan_to_num(chunk_weights, nan=1.0)
        chunk_weights /= chunk_weights.sum(axis=1, keepdims=True)

        if fine_func.ndim > 1:
            for j in range(chunk_weights.shape[0]):
                np.add.at(coarse_func, indices[j],
                          chunk_weights[j][:, None] * fine_func[i + j])
        else:
            for j in range(chunk_weights.shape[0]):
                np.add.at(coarse_func, indices[j],
                          chunk_weights[j] * fine_func[i + j])

        fine_to_coarse[i:chunk_end] = indices
        weights[i:chunk_end] = chunk_weights

    if np.isnan(coarse_func).any():
        print("Fixing NaN values...")
        for i in np.where(np.isnan(coarse_func).any(axis=1))[0]:
            valid_sources = fine_to_coarse == i
            if valid_sources.any():
                coarse_func[i] = np.average(
                    fine_func[valid_sources.any(axis=1)],
                    weights=weights[valid_sources],
                    axis=0
                )
            else:
                coarse_func[i] = np.nanmean(fine_func, axis=0)

    return coarse_func.squeeze(), {
        'indices': fine_to_coarse,
        'weights': weights
    }


def validate_hierarchy(hierarchy):
    print("\nValidating hierarchy structure...")
    issues = 0

    for level, data in hierarchy.levels.items():
        print(f"Checking level {level}: {len(data['vertices'])} vertices")

        max_face_idx = data['faces'].max()
        if max_face_idx >= len(data['vertices']):
            print(f"Error: Faces contain invalid vertex index (max {max_face_idx} >= {len(data['vertices'])})")
            issues += 1

        if data['function'] is not None:
            nan_count = np.isnan(data['function']).sum()
            if nan_count > 0:
                print(f"Warning: Found {nan_count} NaN values")

    for key, mapping in hierarchy.mapping.items():
        if mapping['indices'] is None:
            continue
        max_idx = mapping['indices'].max()
        to_level = int(key.split('_to_')[1][1:])
        if max_idx >= len(hierarchy.levels[to_level]['vertices']):
            print(f"Error: Mapping {key} contains invalid index "
                  f"(max {max_idx} >= {len(hierarchy.levels[to_level]['vertices'])})")
            issues += 1

    print(f"Validation complete. Found {issues} issue(s)")
    return issues == 0


if __name__ == "__main__":
    CONFIG = {
        'lh_surf': 'lh.white',
        'rh_surf': 'rh.white',
        'lh_func': 'lh.func.mgh',
        'rh_func': 'rh.func.mgh',
        'output_dir': 'tpl_onavg_hierarchy'
    }

    verts, faces, func = load_tpl_onavg_data(
        CONFIG['lh_surf'], CONFIG['rh_surf'],
        CONFIG['lh_func'], CONFIG['rh_func']
    )

    hierarchy = create_hierarchical_mesh(
        verts, faces, func,
        reduction_ratios=[0.8, 0.6, 0.4, 0.2, 0.1]
    )

    if validate_hierarchy(hierarchy):
        print("\nSaving hierarchy...")
        hierarchy.save(CONFIG['output_dir'])
        print(f"Successfully saved to {CONFIG['output_dir']}")

        print("\n=== Hierarchy Summary ===")
        for level, data in hierarchy.levels.items():
            res = hierarchy.metadata['level_resolutions'][level]
            print(f"Level {level}: {len(data['vertices'])} vertices, "
                  f"Resolution: {res:.2f} mm, "
                  f"Functional data: {'Yes' if data['function'] is not None else 'No'}")
    else:
        print("\nCritical issues found. Results were not saved.")