import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

class Aperture:
    def __init__(this, display_xwidth: float, display_ywidth: float, steps_x: int, steps_y: int):
        this.interval_x: float = display_xwidth / steps_x
        this.interval_y: float = display_ywidth / steps_y

        this.x1d: np.ndarray = this.interval_x * (np.arange(steps_x) - steps_x // 2) # Floor, integer-like division
        this.y1d: np.ndarray = this.interval_y * (np.arange(steps_y) - steps_y // 2)

        this.x2d: np.ndarray
        this.y2d: np.ndarray
        this.x2d, this.y2d = np.meshgrid(this.x1d, this.y1d)

        this.aperture: np.ndarray = np.zeros((steps_y, steps_x))
        this.airy: bool = False # Flag for plotting; true if a circular aperture is created

    def insert_rectangle(this, center_x: float, center_y: float, width: float, height: float):
        this.aperture += np.select([((this.x2d > (center_x - width / 2)) & (this.x2d < (center_x + width / 2)))
                                          & ((this.y2d > (center_y - height / 2)) & (this.y2d < (center_y + height / 2))), True],
                                   [1, 0])

    def insert_circle(this, center_x: float, center_y: float, radius: float):
        this.aperture += np.select([(this.x2d - center_x) ** 2 + (this.y2d - center_y) ** 2 < radius ** 2, True],
                                   [1, 0])

    def single_slit(this, slit_width: float, slit_height: float):
        this.insert_rectangle(center_x=0, center_y=0, width=slit_width, height=slit_height)

    def double_slit(this, slit_width: float, slit_height: float, distance_between_slits: float):
        this.insert_rectangle(center_x=-distance_between_slits / 2, center_y=0, width=slit_width, height=slit_height)
        this.insert_rectangle(center_x=+distance_between_slits / 2, center_y=0, width=slit_width, height=slit_height)

    def triple_slit(this, slit_width: float, slit_height: float, distance_between_slits: float):
        this.insert_rectangle(center_x=-distance_between_slits, center_y=0, width=slit_width, height=slit_height)
        this.insert_rectangle(center_x=0, center_y=0, width=slit_width, height=slit_height)
        this.insert_rectangle(center_x=+distance_between_slits, center_y=0, width=slit_width, height=slit_height)

    def quadruple_slit(this, slit_width: float, slit_height: float, distance_between_slits: float):
        this.insert_rectangle(center_x=-distance_between_slits / 2 - distance_between_slits, center_y=0, width=slit_width, height=slit_height)
        this.insert_rectangle(center_x=-distance_between_slits / 2, center_y=0, width=slit_width, height=slit_height)
        this.insert_rectangle(center_x=+distance_between_slits / 2, center_y=0, width=slit_width, height=slit_height)
        this.insert_rectangle(center_x=+distance_between_slits / 2 + distance_between_slits, center_y=0, width=slit_width, height=slit_height)

    def circular_slit(this, radius: float):
        this.insert_circle(0, 0, radius)
        this.airy = True

def main():
    # Create the screen
    display_width = 2.8
    display_height = 0.8
    steps_x = 7500
    steps_y = 4500
    aperture = Aperture(display_xwidth = display_width, display_ywidth = display_height, steps_x = steps_x, steps_y = steps_y)

    # Define the aperture and laser beam's parameters
    slit_width = 30 * 1e-3 # 30 microns
    slit_height = 90 * 1e-3 # 90 microns
    distance_between_slits = 130 * 1e-3 # 130 microns
    radius = 30 * 1e-3 # 30 microns
    wavelength = 633 * 1e-6  # 633 nanometers
    wavenumber = 2 * np.pi / wavelength
    distance_to_screen = 2000 # 2 meters

    # Uncomment one more or of these aperture-creation functions to see a diffraction pattern #
    #aperture.single_slit(slit_width, slit_height)
    aperture.double_slit(slit_width, slit_height, distance_between_slits)
    #aperture.triple_slit(slit_width, slit_height, distance_between_slits)
    #aperture.quadruple_slit(slit_width, slit_height, distance_between_slits)
    aperture.circular_slit(radius)

    # Perform the Fourier transformation upon the aperture
    fft_complex = fft2(aperture.aperture * np.exp(1j * wavenumber / (2 * distance_to_screen) * (aperture.x2d ** 2 + aperture.y2d ** 2)))
    fft_complex_shifted = fftshift(fft_complex)
    abs_fft_complex_shifted = np.absolute(fft_complex_shifted)

    # Set up the plots
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(6, 9))
    ax0 = fig.add_subplot(3, 1, 1)
    ax1 = fig.add_subplot(3, 1, 2)
    ax2 = fig.add_subplot(3, 1, 3, sharex=ax1, yticklabels=[])

    # The aperture's plot
    ax0.imshow(aperture.aperture,
            extent=(aperture.x1d[0], aperture.x1d[-1] + aperture.interval_x, aperture.y1d[0], aperture.y1d[-1] + aperture.interval_y),
            cmap='terrain', interpolation='bicubic', aspect='auto')
    if aperture.airy:
        ax0.set_title("Aperture (radius = " + str(int(radius * 1e3)) + " μm)")
    else:
        ax0.set_title("Aperture (slits " + str(int(slit_width * 1e3)) + " μm by " + str(int(slit_height * 1e3))
                      + " μm spaced by " + str(int(distance_between_slits * 1e3)) + " μm)")
    ax0.set_xlabel("x (millimeters)")
    ax0.set_ylabel("y (millimeters)")

    # The diffraction pattern's plot
    dx_screen = distance_to_screen * wavelength / display_width
    dy_screen = distance_to_screen * wavelength / display_height
    x_screen = dx_screen * (np.arange(steps_x) - steps_x // 2)
    y_screen = dy_screen * (np.arange(steps_y) - steps_y // 2)

    ax1.set_title("Diffraction pattern (" + str(distance_to_screen / 1e3) + " m away with λ = " + str(int(wavelength / 1e-6)) + " nm)")
    ax1.set_ylabel("y (millimeters)")
    ax1.set_ylim((-50, 50))

    ax1.imshow(abs_fft_complex_shifted, extent=(x_screen[0], x_screen[-1] + dx_screen, y_screen[0], y_screen[-1] + dy_screen),
               cmap='terrain', interpolation="bicubic", aspect='auto')

    # The horizontal intensity slice's plot
    ax2.set_title("Horizontal intensity slice at y = 0")
    ax2.set_xlabel("x (millimeters)")
    ax2.set_ylabel("Intensity (A.U.)")
    ax2.set_xlim((-100, 100))
    ax2.plot(x_screen, abs_fft_complex_shifted[steps_y // 2] ** 2, linewidth=1.5)

    # Save and display the plots
    plt.tight_layout()
    plt.savefig('diffraction_tie-fighter.svg', dpi=1200, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
