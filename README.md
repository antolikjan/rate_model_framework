Setup:

changing line 1321 in "site-packages\imagen\__init__.py" (linspace)
        time_axis = np.linspace(0.0, duration, duration*sample_rate)

to 
    time_axis = np.linspace(int(0.0), int(duration), int(duration*sample_rate))

is necessary for correct import of Imagen