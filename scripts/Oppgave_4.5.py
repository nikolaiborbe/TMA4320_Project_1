 
 
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
 
 # NN vs FDM
    error = T_pred - T_fdm
    rmse = np.sqrt(np.mean(error**2))
    print(rmse)