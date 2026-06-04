


import numpy as np
a = np.load('sh_linel.npy')      # non-vec (loop version)
b = np.load('sh_linel_vec.npy')  # vec
print('shape:', a.shape, b.shape)
print('dtype:', a.dtype, b.dtype)
print('non-vec max abs:', np.max(np.abs(a)))
print('vec max abs:    ', np.max(np.abs(b)))
print('max abs diff:   ', np.max(np.abs(a - b)))
print('mean abs diff:  ', np.mean(np.abs(a - b)))
print('sum abs diff:   ', np.sum(np.abs(a - b)))
print('rel max diff:   ', np.max(np.abs(a - b)) / np.max(np.abs(a)))

worst = np.unravel_index(np.argmax(np.abs(a - b)), a.shape)
print(f'worst entry: index={worst}')
print(f'  non-vec value = {a[worst]}')
print(f'  vec value     = {b[worst]}')
print(f'  abs diff      = {np.abs(a[worst] - b[worst])}')

# Per-component breakdown
print('\nPer-component max abs diff:')
labels = ['Nx', 'Ny', 'Nxy', 'Mx', 'My', 'Mxy', 'Qx', 'Qy']
for c in range(a.shape[-1]):
    print(f'  {labels[c]:4s}: max_diff={np.max(np.abs(a[..., c] - b[..., c])):.6g}, '
          f'non-vec max={np.max(np.abs(a[..., c])):.6g}')