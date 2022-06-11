import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3

# draw a "comic strip" style rendering of the given sequence of poses
def draw_comic(frames, angles=None, figsize=None, window_size=0.45, dot_size=20, lw=2.5, zcolor=None,cmap='cool_r'):
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    ax.view_init(30, 0)
    shift_size=window_size
    
    ax.set_xlim(-window_size,window_size)
    ax.set_ylim(-window_size,len(frames)*window_size)
    ax.set_zlim(-0.1,0.6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    cm = matplotlib.cm.get_cmap(cmap)
    
    if angles is not None:
        vR = 0.15
        zidx = point_labels.index("CLAV")
        X = frames[:,zidx,0]
        Y = frames[:,zidx,1]
        dX,dY = vR*np.cos(angles), vR*np.sin(angles)
        Z = frames[:,zidx,2]
        #Z = frames[:,2,2]
 
    
    for iframe,frame in enumerate(frames):
        ax.scatter(frame[:,0],
                       frame[:,1]+iframe*shift_size,
                       frame[:,2],
                       alpha=0.3,
                       c=zcolor,
                       cmap=cm,
                       s=dot_size,
                       depthshade=True)
        
        if angles is not None:
            ax.quiver(X[iframe],iframe*shift_size+Y[iframe],Z[iframe],dX[iframe],dY[iframe],0, color='black')
        
        for i,(g1,g2) in enumerate(skeleton_lines):
            g1_idx = [point_labels.index(l) for l in g1]
            g2_idx = [point_labels.index(l) for l in g2]

            if zcolor is not None:
                color = cm(0.5*(zcolor[g1_idx].mean() + zcolor[g2_idx].mean()))
            else:
                color = None

            x1 = np.mean(frame[g1_idx],axis=0)
            x2 = np.mean(frame[g2_idx],axis=0)
            
            ax.plot(np.linspace(x1[0],x2[0],10),
                    np.linspace(x1[1],x2[1],10)+iframe*shift_size,
                    np.linspace(x1[2],x2[2],10),
                    color=color,
                    lw=lw)