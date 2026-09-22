# M0404 Pressure Map

This browser tool displays the two 4x4 M0404 modules as one 4x8 color heatmap.
The numeric 4x8 layout and the local timestamp are shown below the map.

## Run

Use Microsoft Edge or Google Chrome. Web Serial requires a secure browser
context, so serve this directory from localhost instead of opening the HTML file
directly:

```powershell
cd STM32CubeIDE\workspace_1.18.1\09_DUAL_M0404_PARSER\visualizer
python -m http.server 4173 --bind 127.0.0.1
```

Open <http://localhost:4173>, select the board's serial port, and keep the
baud rate at `115200`. Click **Demo** first to verify the color scale without
hardware. The firmware must send lines in this form:

```text
P32,t=123456,v0,v1,...,v31
```

`t` is the MCU `HAL_GetTick()` value in milliseconds. Values are row-major:
each row contains four points from M1 followed by four points from M2. The
firmware sends this compact frame every 50 ms (20 Hz). The parser ignores all
other serial lines, including the diagnostic and `GRID4x8` lines.

## Bluetooth IMU

The page also connects directly to a BWT901BLECL5.0 in Microsoft Edge or Chrome
through Web Bluetooth. Open the page from `http://127.0.0.1:4173/`, close
`WitMotion.exe`, click **Connect IMU**, and select the paired sensor. The page
uses the WIT Bluetooth 5.0 service `FFE5` and notification characteristic `FFE4`;
it parses the 20-byte `55 61` frame and shows roll, pitch, yaw, acceleration,
and gyro values. If the sensor is not configured for automatic streaming, the
page polls the `FFE9` write characteristic for the same registers. IMU frames
show the PC receive timestamp and an MCU timestamp synchronized to the latest
pressure frame.

On each IMU connection, the page sets the WIT bandwidth to 188 Hz and return
rate to 200 Hz, then saves the setting. The IMU status line shows the received
`0x55 0x61` frame rate once per second. The display itself is limited by the
monitor refresh rate. Raw acceleration is shown in `g` and includes gravity;
the `Linear` values subtract gravity estimated from roll and pitch.
Angular acceleration is the filtered derivative of the gyro data and is shown
in `deg/s2`.

Only one application should own the IMU connection at a time. If the sensor is
not listed, pair it in Windows Bluetooth settings first, then refresh the page.
