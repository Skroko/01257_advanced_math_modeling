#%%
import sympy as sp
a = sp.Symbol("a", positive=True, real=True)
b = sp.Symbol("b", positive=True, real=True)

A_b = sp.Matrix([
    [0,-1],
    [a,-1]
])

B_b = sp.Matrix([
    [0,1],
    [0,1-b]
])

O_b = sp.Matrix([
    [0,0],
    [0,0]
])


A3 = sp.BlockMatrix([
    [A_b,B_b,O_b],
    [O_b,A_b,B_b],
    [B_b,O_b,A_b]
    ]).as_explicit()

A2 = sp.BlockMatrix([
    [A_b,B_b],
    [B_b,A_b]
    ]).as_explicit()


#%%
import matplotlib.pyplot as plt
import numpy as np
eigenvals = [i for i in list(A3.eigenvals().keys())]
print(eigenvals[0])


# var_vals = {a:1,b:1}
# eigenvals[1].subs(var_vals).simplify().evalf()

N = 30
min_val, max_val =  1e-3,100
_a_s = np.linspace(min_val,max_val,N)
_b_s = np.linspace(min_val,max_val,N)

aa_s,bb_s = np.meshgrid(_a_s,_b_s)

a_s = aa_s.ravel()
b_s = bb_s.ravel()

bot = np.zeros_like(a_s)


#%%

def get_eigv_from_M(ai,bi):
    np_A3 = np.array(A3.subs({a:ai,b:bi}).tolist(),dtype=np.float32)
    eigenvals = np.linalg.eigvals(np_A3)
    return np.sort(eigenvals)

eigs = get_eigv_from_M(1,2)

#%%

eigs_dict = {f"eig{i}":[] for i in range(len(eigs))}

for ai in _a_s:
    for bi in _b_s:
        eigenvals = get_eigv_from_M(ai,bi)
        for key,val in zip(eigs_dict.keys(),eigenvals):
            eigs_dict[key] += [val]
        

#%%

eig_real = {key: np.real(val).reshape(aa_s.shape) for key, val in eigs_dict.items()}

# Plotting setup
fig, axes = plt.subplots(2, 3, subplot_kw={"projection": "3d"}, figsize=(18, 12))
axes = axes.flatten()

for idx, (key, eig_values) in enumerate(eig_real.items()):
    ax = axes[idx]
    
    # 3D surface plot
    surf = ax.plot_surface(aa_s, bb_s, eig_values.T, cmap='viridis', edgecolor='none')
    ax.set_title(f"{key}")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("Re(eigenvalue)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
# %%

