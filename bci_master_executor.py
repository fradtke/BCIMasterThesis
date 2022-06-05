# Import area 
from learning_procedure_executor import LearningProcedureExecutor 

# Area of settings
# GIST 
from Properties_settings.Data_settings import settings_gist_data
from Properties_settings.Network_settings import network_settings_default

# ISC2020 
from Properties_settings.Data_settings import settings_imagined_speech_c2020
from Properties_settings.Network_settings import network_settings_default


# Data area 
setting_list = [
    # GIST 
    (settings_gist_data,
     network_settings_default),
    # ISC2020 
    (settings_imagined_speech_c2020,
     network_settings_default),
]

# Execution area 
if __name__ == '__main__': 
    for settings_pair in setting_list: 
        LearningProcedureExecutor.conduct_network_learning_procedure_for_data_sample( 
            parameter_settings=settings_pair[0],
            network_settings=settings_pair[1]
        )
