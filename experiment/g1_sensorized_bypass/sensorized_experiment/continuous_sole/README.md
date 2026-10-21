# Continuous Sole Integration

中文总览和图表索引见 `../../../docs/foot_sensing_project_overview_cn.md`。

This experiment is a separate copy of the downloaded G1 walking candidate.
The candidate files under `external/candidates` are not modified.

## Model

`build_continuous_model.py` creates `model/g1_12dof_continuous.xml` and scene
copies under `scenes/`. Original ankle/foot collision geoms are disabled, and
each ankle-roll body receives one thin rectangular sole plate. The plate is a
single continuous collision surface; the 8 x 4 grid is only a virtual sensing
partition used during post-processing.

## Per-cell integration

`continuous_sensor_suite.py` assigns every MuJoCo contact sample on the sole to
the cell containing its contact position and sums its three-dimensional force:

\[
\mathbf F_{ij}=\sum_{k\in A_{ij}}\mathbf f_k\approx\int_{A_{ij}}\mathbf t(\mathbf r)\,dA.
\]

`fx`, `fy`, and `fz` are the integrated cell force components. A cell's
`pressure_avg_pa` is the normal resultant divided by the *full nominal cell
area*, `max(Fz, 0) / A_cell`; it is therefore a cell-average pressure, not the
pressure density over only the active contact patch. `contact_count` records how
many solver contact samples contributed to the cell.

The logger also writes the ankle macro wrench, CoP, ankle IMU, outside-foot
force, and `force_closure_error_n`. The latter compares the sum of all in-grid
cell forces plus outside-foot contacts with the macro contact force.

MuJoCo still supplies discrete solver contact samples even though the collision
surface is continuous. Thus this is a virtual array integral over one plate,
not a claim that each grid cell is a physical MuJoCo geom or that the solver
has directly returned a continuum traction field.

## Run

```text
uv run --no-project --with mujoco --with torch --with numpy python run_continuous_sole_walk.py --scene rough --speed 0.4 --duration 8 --episodes 2 --output outputs/uniform_speed_integrated/rough_04_integrated.csv
```

`analyze_integrated_sole.py` summarizes force-closure and occupancy for the
generated CSV files. The existing `../analyze_sensor_information.py` can also
consume these files for macro-versus-array information analysis.

`analyze_boundary_and_warning.py` performs the three-claim offline study:
macro-state ambiguity, pre-fall warning, and single-foot directional loading
margin. The completed 20-episode rough-terrain study is documented in
`THREE_CLAIMS_REPORT.md`; it does not change or retrain the walking controller.

`analyze_posture_adjustment_examples.py` adds a deliberately small
counterfactual for the next question: can the measurements justify changing
the *current foot attitude* at an already occupied landing point? It freezes a
logged snapshot, keeps the normal load and ankle wrench fixed, and compares
the nominal attitude with virtual +/-3 degree pitch/roll corrections. The
array uses the active-cell support interval; the IMU marks dynamic urgency.
This is not a new controller or a replayed MuJoCo branch. It is an offline
one-step test that reports whether the original attitude is below a selected
margin and whether a sensor-informed correction would move all four
directional margins above it.

Run it on the existing 20-episode data:

```text
uv run --no-project --with numpy --with matplotlib python analyze_posture_adjustment_examples.py outputs/warning_validation/rough_04_20episodes.csv --output-dir outputs/posture_adjustment_demo --examples 2
```

The generated `POSITION_ADJUSTMENT_REPORT.md` contains two concrete rough-
terrain snapshots and figures. The first is the important distributed-sensing
case: the full-sole macro model says to hold, while the active plantar region
has insufficient margin and a roll correction restores it. The second shows
the complementary case in which the macro state already looks dangerous but
does not supply a posture action under the original strategy.
