w 
 
 """
 Evaluer resultatene ved å sammenligne med den numeriske løseren. Er NN i stand til å
lære temperaturfeltet fra sensordataene alene?
Andre ting som kan være relevante å diskutere:
• Hvordan påvirkes resultatene av antall sensorer og mengden støy i målingene?
• Hvordan påvirkes resultatene av nettverksarkitekturen (antall lag og nevroner per lag)?
• Hvordan påvirkes resultatene av antall epoker og parametrene til Adam-algoritmen?

"""

 from project import (
    load_config,
    solve_heat_equation,
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)
from matplotlib.pyplot import plt

t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)
 
 # NN vs FDM
    error = T_pred - T_fdm
    rmse = np.sqrt(np.mean(error**2))
    print(rmse)

# Plott
plt.plot(T_pred, t)
plt.plot(T_fdm, t)
plt.xlabel("Temperature")
plt.ylabel("time t")
plt.show()
