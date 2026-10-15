# -*- coding: utf-8 -*-
"""
Fill-based piecewise pressure optimizer for Abaqus/CAE Python 2.7.

Main idea:
    The forming process is split by die filling fraction. At each filling interval,
    several pressure values are tested from the same initial state. Only candidates
    that reach the target filling fraction are allowed to compete. The winner is the
    candidate with the maximum minimum thickness.

Important implementation details:
    1. The pressure amplitude used by the model is updated explicitly by name.
       By default this script updates Amp_1, because the model load may already
       be attached to Amp_1.
    2. TOP and BOTTOM layers are extracted exactly as in the report parsing workflow:
       all nodal COORD values are sorted by node label; for 9 nodes through thickness,
       TOP = rows 1, 10, 19, ... and BOTTOM = rows 9, 18, 27, ...
    3. Thickness is computed as the shortest distance from each BOTTOM point to the
       nearby TOP polyline segments.
    4. The final selected ODB and the selected pressure path are written to separate
       files.

Run inside Abaqus/CAE:
    abaqus cae noGUI=fill_pressure_optimizer_layer_based.py
"""

from abaqus import *
from abaqusConstants import *
import visualization
import numpy as np
import os
import traceback

# ============================================================
# USER PARAMETERS
# ============================================================

MODEL_NAME = 'Model-1'
STEP_NAME = 'Step-1'

INSTANCE_KEYWORD = 'DETAIL'

# The pressure load in the model must use this amplitude.
# If your model currently has Amp_1 attached to the pressure load, keep this name.
AMPLITUDE_NAME = 'Amp_1'

# Safer option: set PRESSURE_LOAD_NAMES = ['Your pressure load name'].
PRESSURE_LOAD_NAMES = []
BIND_ALL_LOADS_IF_EMPTY = True
SET_LOAD_MAGNITUDE_TO_ONE = True

DIE_AREA = 15995.90

# 'RAW' -> F = area_current / DIE_AREA
# 'NORMALIZED_FROM_INITIAL' -> F = (area_current - area_initial)/(DIE_AREA - area_initial)
FILL_MODE = 'NORMALIZED_FROM_INITIAL'

FILL_STEP = 0.1
# Final target filling. We intentionally stop at 0.99 instead of 1.0,
# because exact 1.0 may be numerically hard to reach in forming/contact problems.
FINAL_FILL_TARGET = 0.95
# Backward-compatible alias used in logs and checks.
MAX_FILL = FINAL_FILL_TARGET
FILL_TOLERANCE = 0.01
STRICT_FILL_TOLERANCE = True

PRESSURE_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

RAMP_TIME = 0.5
STAGE_TIME = 300  # TODO: set required maximum duration of one candidate stage
# IMPORTANT FOR RESTART CONTINUATION:
# In Abaqus restart inside the same step, amplitude times and step end time
# are interpreted in the absolute STEP time, not in a local stage time.
# Therefore for stage k we use:
#   stage_start_time = selected frameValue from previous stage
#   stage_end_time   = stage_start_time + STAGE_TIME
# and the pressure amplitude is defined at these absolute times.
USE_ABSOLUTE_RESTART_TIME = True

# Layer extraction from all nodal COORD rows sorted by node label.
# This reproduces:
#   top_data    = data_lines[0::9]
#   bottom_data = data_lines[8::9]
NODES_THROUGH_THICKNESS = 7
TOP_LAYER_OFFSET = 0
BOTTOM_LAYER_OFFSET = 6
EXPECTED_TOTAL_NODES = None  # set None to disable diagnostic

THICKNESS_SEARCH_WINDOW = 1

# Restart ODBs may contain copied history from the source ODB.
# Such inherited frames must not be used to compare candidate pressures,
# otherwise all candidates can be evaluated at exactly the same old frame.
SKIP_INHERITED_RESTART_HISTORY = True
INHERITED_FRAME_MATCH_TOL = 1.0e-7
MIN_CURRENT_STAGE_ELAPSED_TIME = 1.0e-9

# Interpolate the time at which the target filling is crossed between two
# adjacent ODB frames. The restart still uses the real saved frame, because
# Abaqus can restart only from a written increment. The exported replay
# pressure schedule may use the interpolated crossing time.
USE_INTERPOLATED_TARGET_TIME = True
EXPORT_INTERPOLATED_PRESSURE_SCHEDULE = False
WRITE_RESTART_PRESSURE_HISTORY_CSV = True
PRESSURE_EQUAL_TOL = 1.0e-10

# Important restart-filter rule:
# Frames copied from the source ODB after the restart increment must be skipped
# only when the candidate pressure differs from the pressure active at restart.
# If the candidate pressure is the same as the previous pressure, those frames
# are not an error: they are the physically correct continuation of the same
# pressure path and may already contain the next target filling.
SKIP_INHERITED_ONLY_IF_PRESSURE_CHANGES = True

BASE_JOB_PREFIX = 'Opt_smirnov'
NUM_CPUS = 10
NUM_DOMAINS = 10
USER_SUBROUTINE = r'abaqus_subroutines/smirnov_usdfld.for'

LOG_DIR = r'data/forming/logs/run_01'
LOG_FILE = os.path.join(LOG_DIR, 'fill_pressure_optimizer_log.txt')
SUMMARY_CSV = os.path.join(LOG_DIR, 'fill_pressure_optimizer_summary.csv')
LIVE_INCREMENT_LOG = os.path.join(LOG_DIR, 'fill_pressure_optimizer_increments_live.txt')
BEST_PATH_CSV = os.path.join(LOG_DIR, 'fill_pressure_optimizer_best_path.csv')
PRESSURE_SCHEDULE_CSV = os.path.join(LOG_DIR, 'fill_pressure_optimizer_pressure_schedule.csv')
RESTART_PRESSURE_SCHEDULE_CSV = os.path.join(LOG_DIR, 'fill_pressure_optimizer_restart_pressure_history.csv')
FINAL_RESULT_TXT = os.path.join(LOG_DIR, 'fill_pressure_optimizer_final_result.txt')

USE_RESTART = True

# Use COORD because your .rpt export uses COORD at NODAL position.
COORDINATE_MODE = 'COORD'
ALLOW_U_FALLBACK_IF_COORD_MISSING = True

STATUS_OK = 'OK'
STATUS_NOT_REACHED = 'NOT_REACHED_TARGET_FILL'
STATUS_REACHED_LARGE_ERROR = 'REACHED_LARGE_FILL_ERROR'
STATUS_FAILED = 'FAILED'

REFERENCE_TOP_AREA = None

# ============================================================
# SMALL HELPERS
# ============================================================

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def pressure_to_tag(pressure):
    return str(pressure).replace('.', 'p').replace('-', 'm')


def get_odb_path(job_name):
    return job_name + '.odb'


def make_fill_targets(fill_step, final_fill_target):
    """
    Build filling targets and always include the final target.

    Example:
        fill_step = 0.15, final_fill_target = 0.99
        targets = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.99]

    This avoids stopping at the last multiple of FILL_STEP, e.g. 0.75 or 0.90,
    when the final requested filling is not divisible by the step.
    """
    step = float(fill_step)
    final_target = float(final_fill_target)

    if step <= 0.0:
        raise RuntimeError('FILL_STEP must be positive.')
    if final_target <= 0.0 or final_target > 1.0:
        raise RuntimeError('FINAL_FILL_TARGET must be in the interval (0, 1].')

    targets = []
    value = step
    eps = 1.0e-10

    while value < final_target - eps:
        targets.append(float(value))
        value += step

    # Always append the requested final filling target, even if it is not a
    # multiple of the step. Avoid duplicates if it is already present.
    if not targets or abs(targets[-1] - final_target) > 1.0e-8:
        targets.append(float(final_target))

    return targets


# ============================================================
# ODB INSTANCE AND COORDINATE EXTRACTION
# ============================================================

def find_instance(odb):
    keyword = INSTANCE_KEYWORD.upper()
    for name, inst in odb.rootAssembly.instances.items():
        if keyword in name.upper():
            print('Using instance: %s' % name)
            return inst
    print('ERROR: Instance not found by keyword: %s' % INSTANCE_KEYWORD)
    print('Available instances:')
    for name in odb.rootAssembly.instances.keys():
        print('  %s' % name)
    return None


def get_field_values_by_node_label(frame, instance, field_name, position=NODAL):
    if field_name not in frame.fieldOutputs:
        return None

    field = frame.fieldOutputs[field_name]
    try:
        subset = field.getSubset(region=instance, position=position)
    except Exception:
        subset = field.getSubset(region=instance)

    data = {}
    for v in subset.values:
        if hasattr(v, 'nodeLabel'):
            data[int(v.nodeLabel)] = v.data
    return data


def get_all_node_records_from_coord(frame, instance):
    """
    Read deformed nodal coordinates from COORD field and return:
        [(node_label, x, y), ...]
    sorted by node label.
    """
    coord_by_label = get_field_values_by_node_label(frame, instance, 'COORD', NODAL)
    if coord_by_label is None:
        raise RuntimeError('COORD field output is not available in this frame.')

    records = []
    for label, data in coord_by_label.items():
        records.append((int(label), float(data[0]), float(data[1])))

    records.sort(key=lambda item: item[0])
    return records


def get_all_node_records_from_u_plus_initial(frame, instance):
    u_by_label = get_field_values_by_node_label(frame, instance, 'U', NODAL)
    if u_by_label is None:
        raise RuntimeError('U field output is not available for U_PLUS_INITIAL mode.')

    records = []
    nodes = list(instance.nodes)
    nodes.sort(key=lambda n: n.label)

    for node in nodes:
        label = int(node.label)
        u = u_by_label.get(label, None)
        if u is None:
            continue

        x0 = float(node.coordinates[0])
        y0 = float(node.coordinates[1])
        x = x0 + float(u[0])
        y = y0 + float(u[1])
        records.append((label, x, y))

    records.sort(key=lambda item: item[0])
    return records


def get_all_node_records(frame, instance):
    if COORDINATE_MODE.upper() == 'COORD':
        try:
            return get_all_node_records_from_coord(frame, instance)
        except Exception:
            if not ALLOW_U_FALLBACK_IF_COORD_MISSING:
                raise
            print('WARNING: COORD reading failed. Trying U_PLUS_INITIAL fallback.')
            traceback.print_exc()
            return get_all_node_records_from_u_plus_initial(frame, instance)

    if COORDINATE_MODE.upper() == 'U_PLUS_INITIAL':
        return get_all_node_records_from_u_plus_initial(frame, instance)

    raise RuntimeError('Unknown COORDINATE_MODE: %s' % COORDINATE_MODE)


def extract_top_bottom_layers(frame, instance):
    """
    Extract TOP and BOTTOM layers from all nodal coordinates sorted by node label.

    This reproduces the external parser:
        top    = data_lines[0::9]
        bottom = data_lines[8::9]
    """
    records = get_all_node_records(frame, instance)

    n_records = len(records)
    if EXPECTED_TOTAL_NODES is not None and n_records != EXPECTED_TOTAL_NODES:
        print('WARNING: number of nodal COORD records = %d, expected %d' %
              (n_records, EXPECTED_TOTAL_NODES))

    if n_records < NODES_THROUGH_THICKNESS:
        raise RuntimeError('Too few nodes for layer extraction: %d' % n_records)

    if n_records % NODES_THROUGH_THICKNESS != 0:
        print('WARNING: number of nodes %d is not divisible by NODES_THROUGH_THICKNESS=%d' %
              (n_records, NODES_THROUGH_THICKNESS))

    top_records = records[TOP_LAYER_OFFSET::NODES_THROUGH_THICKNESS]
    bottom_records = records[BOTTOM_LAYER_OFFSET::NODES_THROUGH_THICKNESS]

    top_labels = [r[0] for r in top_records]
    bottom_labels = [r[0] for r in bottom_records]

    top_points = np.array([[r[1], r[2]] for r in top_records], dtype=float)
    bottom_points = np.array([[r[1], r[2]] for r in bottom_records], dtype=float)

    # Sort both layers by X for area and segment-distance calculations.
    top_order = np.argsort(top_points[:, 0])
    bottom_order = np.argsort(bottom_points[:, 0])

    top_points = top_points[top_order]
    bottom_points = bottom_points[bottom_order]
    top_labels = [top_labels[int(i)] for i in top_order]
    bottom_labels = [bottom_labels[int(i)] for i in bottom_order]

    return top_points, bottom_points, top_labels, bottom_labels


# ============================================================
# GEOMETRY: FILLING AND THICKNESS
# ============================================================

def area_under_top(top_points):
    if top_points is None or len(top_points) < 2:
        return 0.0

    pts = np.array(top_points, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]

    x = pts[:, 0]
    y = pts[:, 1]

    area = float(np.trapz(y, x))
    if area < 0.0:
        area = abs(area)
    return area


def filling_fraction_from_area(area_current):
    global REFERENCE_TOP_AREA

    if FILL_MODE.upper() == 'RAW':
        fill = float(area_current) / float(DIE_AREA)

    elif FILL_MODE.upper() == 'NORMALIZED_FROM_INITIAL':
        if REFERENCE_TOP_AREA is None:
            REFERENCE_TOP_AREA = float(area_current)
            print('REFERENCE_TOP_AREA was set to %.8f' % REFERENCE_TOP_AREA)

        denom = float(DIE_AREA) - float(REFERENCE_TOP_AREA)
        if abs(denom) < 1.0e-12:
            raise RuntimeError('Invalid normalization: DIE_AREA is too close to REFERENCE_TOP_AREA.')

        fill = (float(area_current) - float(REFERENCE_TOP_AREA)) / denom

    else:
        raise RuntimeError('Unknown FILL_MODE: %s' % FILL_MODE)

    if fill < 0.0:
        fill = 0.0
    if fill > 1.0:
        fill = 1.0
    return float(fill)


def compute_fill(top_points):
    area = area_under_top(top_points)
    fill = filling_fraction_from_area(area)
    return fill, area


def distance_point_to_segment(P, A, B):
    AB = B - A
    denom = float(np.dot(AB, AB))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(P - A))

    t = float(np.dot(P - A, AB) / denom)
    if t <= 0.0:
        closest = A
    elif t >= 1.0:
        closest = B
    else:
        closest = A + t * AB

    return float(np.linalg.norm(P - closest))


def min_distance_to_top(bottom_points, top_points, bottom_labels=None, n=1):
    N = len(bottom_points)
    M = len(top_points)
    dists = np.zeros(N, dtype=float)

    for i in range(N):
        best = 1.0e100
        P = bottom_points[i]

        idx_start = max(0, i - n)
        idx_end = min(M, i + n + 1)
        for k in range(idx_start, idx_end):
            dist = float(np.linalg.norm(P - top_points[k]))
            if dist < best:
                best = dist

        j_start = max(0, i - n)
        j_end = min(M - 2, i + n - 1)
        for j in range(j_start, j_end + 1):
            dist = distance_point_to_segment(P, top_points[j], top_points[j + 1])
            if dist < best:
                best = dist

        dists[i] = best

    min_index = int(np.argmin(dists))
    min_label = -1
    if bottom_labels is not None and min_index < len(bottom_labels):
        min_label = int(bottom_labels[min_index])

    return (
        float(np.min(dists)),
        float(np.mean(dists)),
        float(np.max(dists)),
        dists,
        min_index,
        min_label,
        float(bottom_points[min_index, 0]),
        float(bottom_points[min_index, 1])
    )


# ============================================================
# MODEL AND JOB HELPERS
# ============================================================

def ensure_restart_output_keyword(model_name):
    model = mdb.models[model_name]
    kb = model.keywordBlock
    kb.synchVersions(storeNodesAndElements=False)

    text = '\n'.join(kb.sieBlocks)
    if '*RESTART' in text.upper():
        print('Restart keyword already exists in the model.')
        return

    insert_index = None
    for i, block in enumerate(kb.sieBlocks):
        if block.strip().upper().startswith('*STEP'):
            insert_index = i + 1
            break

    restart_card = '*Restart, write, frequency=1'
    if insert_index is None:
        print('WARNING: *Step card was not found. Restart card was not inserted.')
    else:
        kb.insert(insert_index, restart_card)
        print('Inserted restart output keyword: %s' % restart_card)


def get_stage_start_time(restart_source):
    """
    Return absolute step time at which the current candidate stage starts.

    For the first stage it is 0. For restart stages it is the frameValue
    of the selected frame from the previous stage.
    """
    if restart_source is None:
        return 0.0
    return float(restart_source.get('frame_value', 0.0))


def get_stage_start_pressure(restart_source):
    """
    Return the pressure value that was active in the selected restart frame.

    This value is important for restart stability. If the new amplitude starts
    from zero at the restart time, Abaqus sees an instantaneous unloading from
    the previous equilibrium state. This often causes convergence failure in
    the first restarted increment. Therefore every new candidate stage ramps
    from the pressure that was active at the selected previous frame to the
    tested pressure value.
    """
    if restart_source is None:
        return 0.0
    if 'pressure_at_frame' in restart_source:
        return float(restart_source['pressure_at_frame'])
    if 'selected_pressure' in restart_source:
        return float(restart_source['selected_pressure'])
    return 0.0


def get_stage_end_time(restart_source):
    return get_stage_start_time(restart_source) + float(STAGE_TIME)


def pressure_value_at_time(stage_start_time, stage_start_pressure, target_pressure, frame_time):
    """
    Pressure value corresponding to the piecewise-linear amplitude used
    in one candidate stage.

    The load changes from stage_start_pressure to target_pressure during
    RAMP_TIME and then remains equal to target_pressure.
    """
    t0 = float(stage_start_time)
    p0 = float(stage_start_pressure)
    p1 = float(target_pressure)
    t = float(frame_time)

    if float(RAMP_TIME) <= 0.0:
        return p1
    if t <= t0:
        return p0
    if t >= t0 + float(RAMP_TIME):
        return p1

    alpha = (t - t0) / float(RAMP_TIME)
    return p0 + alpha * (p1 - p0)


def get_schedule_start_time(restart_source):
    """
    Monotonic physical/process time used only for the exported pressure schedule.

    Important: in some Abaqus restart ODB files frame.frameValue may be local
    time measured from the beginning of the restarted analysis, while in other
    cases it may be absolute step time. Therefore the exported amplitude table
    must not rely directly on raw frame.frameValue. Instead, we accumulate the
    elapsed time of selected stages in schedule_time.
    """
    if restart_source is None:
        return 0.0
    return float(restart_source.get('schedule_time', 0.0))


def compute_frame_elapsed_time(restart_source, frame_value):
    """
    Return elapsed time inside the current candidate stage.

    If frame_value is absolute step time, elapsed = frame_value - stage_start_time.
    If frame_value is local restart time and becomes smaller than stage_start_time,
    elapsed = frame_value.

    This prevents non-monotonic pressure schedules such as 6 -> 2 -> 8.
    """
    fv = float(frame_value)
    if restart_source is None:
        return max(0.0, fv)

    local_start = get_stage_start_time(restart_source)

    if fv >= local_start - 1.0e-8:
        return max(0.0, fv - local_start)

    # Restart ODB appears to use local time from the beginning of this restart job.
    return max(0.0, fv)


def pressure_value_after_elapsed(stage_start_pressure, target_pressure, elapsed_time):
    """
    Pressure value after elapsed_time from the beginning of the current stage.

    The selected pressure path ramps linearly from stage_start_pressure to
    target_pressure during RAMP_TIME and then remains constant.
    """
    p0 = float(stage_start_pressure)
    p1 = float(target_pressure)
    dt = float(elapsed_time)

    if float(RAMP_TIME) <= 0.0:
        return p1
    if dt <= 0.0:
        return p0
    if dt >= float(RAMP_TIME):
        return p1

    alpha = dt / float(RAMP_TIME)
    return p0 + alpha * (p1 - p0)


def clean_amplitude_data(data):
    """
    Clean pressure-time table before passing it to Abaqus TabularAmplitude.

    Rules:
        1. Times must be non-decreasing.
        2. If several points have the same time, keep only the last one.
        3. Convert all values to float.

    Abaqus does not like duplicated time values in an amplitude table, while
    restart boundaries naturally create such duplicates.
    """
    cleaned = []
    eps = 1.0e-8

    for item in data:
        if item is None or len(item) < 2:
            continue

        t = float(item[0])
        p = float(item[1])

        if cleaned and t < cleaned[-1][0] - eps:
            print('WARNING: amplitude point with decreasing time was skipped: t=%.8f, previous t=%.8f' %
                  (t, cleaned[-1][0]))
            continue

        if cleaned and abs(t - cleaned[-1][0]) <= eps:
            cleaned[-1] = (t, p)
        else:
            cleaned.append((t, p))

    if len(cleaned) < 2:
        raise RuntimeError('Amplitude table must contain at least two points.')

    return tuple(cleaned)


def get_pressure_history(restart_source):
    """
    Return full selected pressure history up to the current restart state.

    This history is expressed in absolute Abaqus STEP time. It is different
    from local elapsed time. The same history is used:
        1. inside every restart candidate job,
        2. in the final exported pressure schedule.

    This is necessary if we want the restart optimization chain and the final
    continuous replay calculation to use exactly the same pressure-time law.
    """
    if restart_source is None:
        return []

    history = restart_source.get('pressure_history', [])
    result = []
    for item in history:
        if item is None or len(item) < 2:
            continue
        result.append((float(item[0]), float(item[1])))

    return result


def build_candidate_pressure_history(pressure, restart_source=None):
    """
    Build full amplitude data for the current candidate.

    Earlier versions defined only the current stage segment:
        (t0, p0), (t0 + ramp, p_candidate), (t0 + STAGE_TIME, p_candidate)

    That is not ideal for restart/replay equivalence. Now every restart
    candidate receives the complete pressure history from t=0 up to the
    selected restart frame, plus the new candidate segment.

    All times are absolute STEP times.
    """
    p1 = float(pressure)

    if restart_source is None:
        t0 = 0.0
        t_end = float(STAGE_TIME)

        if float(RAMP_TIME) <= 0.0:
            data = [(0.0, p1), (t_end, p1)]
        else:
            data = [(0.0, 0.0), (float(RAMP_TIME), p1), (t_end, p1)]

        return clean_amplitude_data(data)

    t0 = get_stage_start_time(restart_source)
    p0 = get_stage_start_pressure(restart_source)
    t_ramp = t0 + float(RAMP_TIME)
    t_end = t0 + float(STAGE_TIME)

    data = []
    eps = 1.0e-8

    # Keep the complete selected history up to the restart time.
    # Points after t0 are not part of the selected path and must not be reused.
    for t, p in get_pressure_history(restart_source):
        if float(t) <= t0 + eps:
            data.append((float(t), float(p)))

    # Ensure that the table contains an exact point at the restart time.
    # This avoids an unintended pressure jump at the first restarted increment.
    if data and abs(float(data[-1][0]) - t0) <= eps:
        data[-1] = (t0, p0)
    else:
        data.append((t0, p0))

    # Add the candidate segment. If the pressure does not change, do not add
    # an artificial t0 + RAMP_TIME point with the same pressure. It is harmless
    # physically, but it complicates comparisons with continuous replay runs.
    if abs(p1 - p0) <= float(PRESSURE_EQUAL_TOL):
        data.append((t_end, p1))
    elif float(RAMP_TIME) <= 0.0:
        data.append((t0, p1))
        data.append((t_end, p1))
    else:
        data.append((t_ramp, p1))
        data.append((t_end, p1))

    return clean_amplitude_data(data)


def update_pressure_amplitude(pressure, restart_source=None):
    """
    Define pressure amplitude for the current candidate stage.

    Important for restart/continuous-replay equivalence:
        the amplitude table is not only the local current-stage segment.
        It contains the full selected pressure history from t=0 plus the
        tested current candidate segment.

    This makes the pressure law seen by Abaqus in restart jobs as close as
    possible to the pressure law used later in one continuous verification run.
    """
    model = mdb.models[MODEL_NAME]

    data = build_candidate_pressure_history(pressure, restart_source)

    if AMPLITUDE_NAME in model.amplitudes:
        del model.amplitudes[AMPLITUDE_NAME]

    model.TabularAmplitude(
        name=AMPLITUDE_NAME,
        timeSpan=STEP,
        data=data
    )

    print('Amplitude %s was set with full pressure history: points=%d, first=(%.6f, %.6f), last=(%.6f, %.6f)' %
          (AMPLITUDE_NAME, len(data), data[0][0], data[0][1], data[-1][0], data[-1][1]))

def bind_pressure_load_to_amplitude():
    model = mdb.models[MODEL_NAME]

    if PRESSURE_LOAD_NAMES:
        names = PRESSURE_LOAD_NAMES[:]
    elif BIND_ALL_LOADS_IF_EMPTY:
        names = list(model.loads.keys())
    else:
        names = []

    if not names:
        print('WARNING: No pressure load names specified. The script only updates amplitude %s.' %
              AMPLITUDE_NAME)
        return

    for lname in names:
        if lname not in model.loads:
            print('WARNING: load not found: %s' % lname)
            continue

        load = model.loads[lname]
        ok_any = False

        try:
            load.setValues(amplitude=AMPLITUDE_NAME)
            print('Load %s was bound to amplitude %s.' % (lname, AMPLITUDE_NAME))
            ok_any = True
        except Exception:
            print('WARNING: load %s does not accept amplitude=... or update failed.' % lname)

        if SET_LOAD_MAGNITUDE_TO_ONE:
            try:
                load.setValues(magnitude=1.0)
                print('Load %s magnitude was set to 1.0.' % lname)
                ok_any = True
            except Exception:
                print('WARNING: load %s does not accept magnitude=... or update failed.' % lname)

        if not ok_any:
            print('WARNING: load %s was not modified.' % lname)


def update_step_time_period(restart_source=None):
    """
    Set the absolute end time of the Abaqus step.

    In restart continuation inside the same step, timePeriod is treated as the
    final absolute STEP time. It is not an additional local duration.
    Therefore, after restarting from frameValue=t0, we must set:
        timePeriod = t0 + STAGE_TIME
    """
    model = mdb.models[MODEL_NAME]

    if STEP_NAME not in model.steps:
        raise RuntimeError('Step not found in model: %s' % STEP_NAME)

    end_time = get_stage_end_time(restart_source)
    model.steps[STEP_NAME].setValues(timePeriod=float(end_time))
    print('Step time period was set to absolute end time %.6f' % end_time)


def setup_restart_from_previous(prev_job_name, prev_step_name, prev_increment):
    if not USE_RESTART or prev_job_name is None:
        return

    model = mdb.models[MODEL_NAME]
    model.setValues(
        restartJob=prev_job_name,
        restartStep=prev_step_name,
        restartIncrement=int(prev_increment)
    )

    print('Restart source: job=%s, step=%s, increment=%d' %
          (prev_job_name, prev_step_name, int(prev_increment)))


def clear_restart_settings():
    model = mdb.models[MODEL_NAME]
    try:
        model.setValues(restartJob='', restartStep='', restartIncrement=0)
        print('Restart settings were cleared.')
    except Exception:
        pass


def create_form_job(job_name):
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    mdb.Job(
        name=job_name,
        model=MODEL_NAME,
        type=ANALYSIS,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        multiprocessingMode=DEFAULT,
        numCpus=NUM_CPUS,
        numDomains=NUM_DOMAINS,
        userSubroutine=USER_SUBROUTINE
    )


def run_job(job_name, pressure, restart_source):
    update_pressure_amplitude(pressure, restart_source)
    bind_pressure_load_to_amplitude()
    update_step_time_period(restart_source)

    if restart_source is None:
        clear_restart_settings()
    else:
        setup_restart_from_previous(
            restart_source['job_name'],
            restart_source['step_name'],
            restart_source['increment_number']
        )

    create_form_job(job_name)
    job = mdb.jobs[job_name]

    print('Submitting job: %s, pressure = %.6f' % (job_name, pressure))
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()

    if job.status in [ABORTED, TERMINATED]:
        print('JOB FAILED: %s' % job_name)
        return False

    return True


# ============================================================
# LOGGING
# ============================================================

def init_logs():
    ensure_dir(LOG_DIR)

    with open(LOG_FILE, 'w') as f:
        f.write('FILL PRESSURE OPTIMIZER LOG\n')
        f.write('=' * 100 + '\n')
        f.write('MODEL_NAME = %s\n' % MODEL_NAME)
        f.write('STEP_NAME = %s\n' % STEP_NAME)
        f.write('INSTANCE_KEYWORD = %s\n' % INSTANCE_KEYWORD)
        f.write('AMPLITUDE_NAME = %s\n' % AMPLITUDE_NAME)
        f.write('PRESSURE_LOAD_NAMES = %s\n' % str(PRESSURE_LOAD_NAMES))
        f.write('BIND_ALL_LOADS_IF_EMPTY = %s\n' % str(BIND_ALL_LOADS_IF_EMPTY))
        f.write('SET_LOAD_MAGNITUDE_TO_ONE = %s\n' % str(SET_LOAD_MAGNITUDE_TO_ONE))
        f.write('DIE_AREA = %.10f\n' % DIE_AREA)
        f.write('FILL_MODE = %s\n' % FILL_MODE)
        f.write('FILL_STEP = %.6f\n' % FILL_STEP)
        f.write('FINAL_FILL_TARGET = %.6f\n' % FINAL_FILL_TARGET)
        f.write('MAX_FILL = %.6f\n' % MAX_FILL)
        f.write('PRESSURE_VALUES = %s\n' % str(PRESSURE_VALUES))
        f.write('RAMP_TIME = %.6f\n' % RAMP_TIME)
        f.write('STAGE_TIME = %.6f\n' % STAGE_TIME)
        f.write('USE_ABSOLUTE_RESTART_TIME = %s\n' % str(USE_ABSOLUTE_RESTART_TIME))
        f.write('COORDINATE_MODE = %s\n' % COORDINATE_MODE)
        f.write('NODES_THROUGH_THICKNESS = %d\n' % NODES_THROUGH_THICKNESS)
        f.write('TOP_LAYER_OFFSET = %d\n' % TOP_LAYER_OFFSET)
        f.write('BOTTOM_LAYER_OFFSET = %d\n' % BOTTOM_LAYER_OFFSET)
        f.write('SKIP_INHERITED_RESTART_HISTORY = %s\n' % str(SKIP_INHERITED_RESTART_HISTORY))
        f.write('INHERITED_FRAME_MATCH_TOL = %.3e\n' % INHERITED_FRAME_MATCH_TOL)
        f.write('MIN_CURRENT_STAGE_ELAPSED_TIME = %.3e\n' % MIN_CURRENT_STAGE_ELAPSED_TIME)
        f.write('USE_INTERPOLATED_TARGET_TIME = %s\n' % str(USE_INTERPOLATED_TARGET_TIME))
        f.write('EXPORT_INTERPOLATED_PRESSURE_SCHEDULE = %s\n' % str(EXPORT_INTERPOLATED_PRESSURE_SCHEDULE))
        f.write('WRITE_RESTART_PRESSURE_HISTORY_CSV = %s\n' % str(WRITE_RESTART_PRESSURE_HISTORY_CSV))
        f.write('PRESSURE_EQUAL_TOL = %.3e\n' % PRESSURE_EQUAL_TOL)
        f.write('SKIP_INHERITED_ONLY_IF_PRESSURE_CHANGES = %s\n' % str(SKIP_INHERITED_ONLY_IF_PRESSURE_CHANGES))
        f.write('\n')

    with open(SUMMARY_CSV, 'w') as f:
        f.write('stage,target_fill,pressure,job_name,odb_name,status,is_eligible,')
        f.write('frame_index,increment_number,frame_value,top_area,fill,fill_error,max_reached_fill,')
        f.write('min_thickness,mean_thickness,max_thickness,')
        f.write('min_thickness_node_index,min_thickness_node_label,min_thickness_x,min_thickness_y,is_best\n')

    with open(BEST_PATH_CSV, 'w') as f:
        f.write('stage,target_fill,selected_pressure,selected_job,selected_odb,status,')
        f.write('frame_index,increment_number,frame_value,elapsed_time,schedule_time,crossing_time,crossing_fill,crossing_alpha,top_area,fill,min_thickness,pressure_at_frame,pressure_at_crossing\n')

    with open(PRESSURE_SCHEDULE_CSV, 'w') as f:
        f.write('time,pressure\n')

    if WRITE_RESTART_PRESSURE_HISTORY_CSV:
        with open(RESTART_PRESSURE_SCHEDULE_CSV, 'w') as f:
            f.write('time,pressure\n')

    with open(LIVE_INCREMENT_LOG, 'w') as f:
        f.write('LIVE INCREMENT FILLING LOG\n')
        f.write('=' * 120 + '\n')
        f.write('Columns: stage | pressure | job | frame_index | increment | time | top_area | target_fill | fill | fill_error | min_thickness\n')
        f.write('=' * 120 + '\n')

    with open(FINAL_RESULT_TXT, 'w') as f:
        f.write('Final result is not available yet.\n')


def append_live_increment_log(stage, pressure, job_name, frame_index, increment_number,
                              frame_value, top_area, target_fill, fill, min_thickness):
    fill_error = abs(float(fill) - float(target_fill))

    if min_thickness is None:
        min_t = -1.0
    else:
        min_t = float(min_thickness)

    with open(LIVE_INCREMENT_LOG, 'a') as f:
        f.write('%d | %.8f | %s | %d | %d | %.8f | %.8f | %.8f | %.8f | %.8f | %.8f\n' % (
            int(stage),
            float(pressure),
            job_name,
            int(frame_index),
            int(increment_number),
            float(frame_value),
            float(top_area),
            float(target_fill),
            float(fill),
            float(fill_error),
            float(min_t)
        ))
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def append_candidate_to_logs(stage, pressure, result, is_best):
    with open(LOG_FILE, 'a') as f:
        f.write('Stage %d | pressure %.6f | job %s\n' % (stage, pressure, result['job_name']))
        f.write('  status            = %s\n' % result['status'])
        f.write('  is eligible       = %s\n' % str(bool(result['is_eligible'])))
        f.write('  target fill       = %.6f\n' % result['target_fill'])
        f.write('  actual fill       = %.6f\n' % result['fill'])
        f.write('  fill error        = %.6f\n' % result['fill_error'])
        f.write('  max reached fill  = %.6f\n' % result['max_reached_fill'])
        f.write('  top area          = %.6f\n' % result['top_area'])
        f.write('  frame index       = %d\n' % result['frame_index'])
        f.write('  increment number  = %d\n' % result['increment_number'])
        f.write('  frame value       = %.6f\n' % result['frame_value'])
        f.write('  min thickness     = %.6f\n' % result['min_thickness'])
        f.write('  mean thickness    = %.6f\n' % result['mean_thickness'])
        f.write('  max thickness     = %.6f\n' % result['max_thickness'])
        f.write('  min thickness node index = %d\n' % result['min_thickness_node_index'])
        f.write('  min thickness node label = %d\n' % result['min_thickness_node_label'])
        f.write('  min thickness point      = (%.6f, %.6f)\n' %
                (result['min_thickness_x'], result['min_thickness_y']))
        f.write('  pressure at selected frame = %.6f\n' % result.get('pressure_at_frame', -1.0))
        f.write('  elapsed time in stage      = %.6f\n' % result.get('elapsed_time', -1.0))
        f.write('  schedule time              = %.6f\n' % result.get('schedule_time', -1.0))
        f.write('  crossing time             = %.6f\n' % result.get('crossing_time', result.get('frame_value', -1.0)))
        f.write('  crossing fill             = %.6f\n' % result.get('crossing_fill', result.get('fill', -1.0)))
        f.write('  crossing alpha            = %.6f\n' % result.get('crossing_alpha', -1.0))
        f.write('  pressure at crossing      = %.6f\n' % result.get('pressure_at_crossing', result.get('pressure_at_frame', -1.0)))
        f.write('  BEST              = %s\n' % str(bool(is_best)))
        f.write('-' * 100 + '\n')

    with open(SUMMARY_CSV, 'a') as f:
        f.write('%d,%.8f,%.8f,%s,%s,%s,%d,' % (
            int(stage),
            float(result['target_fill']),
            float(pressure),
            result['job_name'],
            result['odb_name'],
            result['status'],
            1 if result['is_eligible'] else 0
        ))
        f.write('%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,' % (
            int(result['frame_index']),
            int(result['increment_number']),
            float(result['frame_value']),
            float(result['top_area']),
            float(result['fill']),
            float(result['fill_error']),
            float(result['max_reached_fill'])
        ))
        f.write('%.8f,%.8f,%.8f,' % (
            float(result['min_thickness']),
            float(result['mean_thickness']),
            float(result['max_thickness'])
        ))
        f.write('%d,%d,%.8f,%.8f,%d\n' % (
            int(result['min_thickness_node_index']),
            int(result['min_thickness_node_label']),
            float(result['min_thickness_x']),
            float(result['min_thickness_y']),
            1 if is_best else 0
        ))


def append_best_path(stage, pressure, result):
    with open(BEST_PATH_CSV, 'a') as f:
        f.write('%d,%.8f,%.8f,%s,%s,%s,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % (
            int(stage),
            float(result['target_fill']),
            float(pressure),
            result['job_name'],
            result['odb_name'],
            result['status'],
            int(result['frame_index']),
            int(result['increment_number']),
            float(result['frame_value']),
            float(result.get('elapsed_time', -1.0)),
            float(result.get('schedule_time', -1.0)),
            float(result.get('crossing_time', result['frame_value'])),
            float(result.get('crossing_fill', result['fill'])),
            float(result.get('crossing_alpha', 1.0)),
            float(result['top_area']),
            float(result['fill']),
            float(result['min_thickness']),
            float(result.get('pressure_at_frame', -1.0)),
            float(result.get('pressure_at_crossing', result.get('pressure_at_frame', -1.0)))
        ))


def append_pressure_schedule_point(schedule_points, time_value, pressure_value):
    """
    Add one point to the final pressure schedule.

    The final schedule must be a clean Abaqus amplitude table:
        time, pressure

    If the new point has the same time as the last point, the last point is
    replaced. This prevents duplicate time rows at restart boundaries.

    A decreasing time is never written. If it appears, this means the raw ODB
    frameValue was local restart time; schedule_time/elapsed_time conversion
    must be used before calling this function.
    """
    t = float(time_value)
    p = float(pressure_value)

    if schedule_points:
        last_t, last_p = schedule_points[-1]

        if t < float(last_t) - 1.0e-8:
            print('WARNING: non-monotonic pressure schedule point was corrected: %.8f -> %.8f' %
                  (t, float(last_t)))
            t = float(last_t)

        if abs(float(last_t) - t) <= 1.0e-8:
            schedule_points[-1] = (t, p)
            return

    schedule_points.append((t, p))


def extend_pressure_schedule(schedule_points, selected_pressure, result, restart_source,
                             use_interpolated_time=False):
    """
    Append the selected stage segment to a pressure schedule.

    Two schedules may be useful:
        1. restart pressure history: uses the real saved frame time, because
           Abaqus restarts from this frame;
        2. diagnostic interpolated schedule: may use the interpolated
           target-crossing time. This is not the restart-equivalent replay law.

    If selected_pressure equals the stage-start pressure, no artificial ramp
    point at t0 + RAMP_TIME is inserted.
    """
    eps = 1.0e-8

    if use_interpolated_time and EXPORT_INTERPOLATED_PRESSURE_SCHEDULE:
        if schedule_points:
            t0 = float(schedule_points[-1][0])
            p0 = float(schedule_points[-1][1])
        else:
            t0 = get_stage_start_time(restart_source)
            p0 = get_stage_start_pressure(restart_source)
        t_end = float(result.get('crossing_time', result.get('frame_value', t0)))
        p_end = float(result.get('pressure_at_crossing', result.get('pressure_at_frame', selected_pressure)))
    else:
        t0 = get_stage_start_time(restart_source)
        p0 = get_stage_start_pressure(restart_source)
        t_end = float(result.get('frame_value', result.get('schedule_time', t0)))
        p_end = float(result.get('pressure_at_frame', selected_pressure))

    p_target = float(selected_pressure)
    t_ramp_end = t0 + float(RAMP_TIME)

    base = []
    for t, p in schedule_points:
        if float(t) <= t0 + eps:
            base.append((float(t), float(p)))

    if base and abs(float(base[-1][0]) - t0) <= eps:
        base[-1] = (t0, p0)
    else:
        base.append((t0, p0))

    if t_end < t0 - eps:
        print('WARNING: selected schedule end time is smaller than segment start time: t_end=%.8f, t0=%.8f. Using t0.' %
              (t_end, t0))
        t_end = t0

    if abs(t_end - t0) <= eps:
        base.append((t_end, p_end))
        schedule_points[:] = list(clean_amplitude_data(base))
        return

    if abs(p_target - p0) <= float(PRESSURE_EQUAL_TOL):
        base.append((t_end, p_target))
    elif float(RAMP_TIME) <= 0.0:
        base.append((t0, p_target))
        base.append((t_end, p_target))
    else:
        if t_end < t_ramp_end - eps:
            base.append((t_end, p_end))
        else:
            base.append((t_ramp_end, p_target))
            base.append((t_end, p_target))

    schedule_points[:] = list(clean_amplitude_data(base))


def write_pressure_schedule_csv(schedule_points, path=None):
    """Write pressure-time table that can be used as Abaqus amplitude data."""
    if path is None:
        path = PRESSURE_SCHEDULE_CSV
    with open(path, 'w') as f:
        f.write('time,pressure\n')
        for t, p in schedule_points:
            f.write('%.8f,%.8f\n' % (float(t), float(p)))



def write_final_result(result, pressure):
    with open(FINAL_RESULT_TXT, 'w') as f:
        f.write('FINAL SELECTED RESULT\n')
        f.write('=' * 80 + '\n')
        f.write('Final ODB: %s\n' % result['odb_name'])
        f.write('Final job: %s\n' % result['job_name'])
        f.write('Final pressure: %.8f\n' % float(pressure))
        f.write('Final step: %s\n' % result['step_name'])
        f.write('Final frame index: %d\n' % int(result['frame_index']))
        f.write('Final increment number: %d\n' % int(result['increment_number']))
        f.write('Final frame value: %.8f\n' % float(result['frame_value']))
        f.write('Final target fill: %.8f\n' % float(result['target_fill']))
        f.write('Final actual fill: %.8f\n' % float(result['fill']))
        f.write('Final top area: %.8f\n' % float(result['top_area']))
        f.write('Final min thickness: %.8f\n' % float(result['min_thickness']))
        f.write('Final pressure at selected frame: %.8f\n' % float(result.get('pressure_at_frame', -1.0)))
        f.write('Final elapsed time in stage: %.8f\n' % float(result.get('elapsed_time', -1.0)))
        f.write('Final schedule time: %.8f\n' % float(result.get('schedule_time', -1.0)))
        f.write('Final interpolated crossing time: %.8f\n' % float(result.get('crossing_time', result.get('frame_value', -1.0))))
        f.write('Final crossing fill: %.8f\n' % float(result.get('crossing_fill', result.get('fill', -1.0))))
        f.write('Final crossing alpha: %.8f\n' % float(result.get('crossing_alpha', -1.0)))
        f.write('Final pressure at crossing: %.8f\n' % float(result.get('pressure_at_crossing', result.get('pressure_at_frame', -1.0))))
        f.write('Final status: %s\n' % result['status'])


# ============================================================
# ODB POSTPROCESSING
# ============================================================

def make_result_item(job_name, idx, frame, target_fill, fill, top_area,
                     max_reached_fill, min_data, status, is_eligible,
                     pressure_at_frame=None, elapsed_time=None, schedule_time=None,
                     crossing_time=None, crossing_fill=None, crossing_alpha=None,
                     pressure_at_crossing=None):
    (min_thick, mean_thick, max_thick, all_thick,
     min_index, min_label, min_x, min_y) = min_data

    if crossing_time is None:
        crossing_time = float(frame.frameValue)
    if crossing_fill is None:
        crossing_fill = float(fill)
    if crossing_alpha is None:
        crossing_alpha = 1.0
    if pressure_at_crossing is None:
        pressure_at_crossing = pressure_at_frame

    return {
        'job_name': job_name,
        'odb_name': get_odb_path(job_name),
        'step_name': STEP_NAME,
        'frame_index': int(idx),
        'increment_number': int(frame.incrementNumber),
        'frame_value': float(frame.frameValue),
        'target_fill': float(target_fill),
        'top_area': float(top_area),
        'fill': float(fill),
        'fill_error': float(abs(fill - target_fill)),
        'max_reached_fill': float(max_reached_fill),
        'min_thickness': float(min_thick),
        'mean_thickness': float(mean_thick),
        'max_thickness': float(max_thick),
        'min_thickness_node_index': int(min_index),
        'min_thickness_node_label': int(min_label),
        'min_thickness_x': float(min_x),
        'min_thickness_y': float(min_y),
        'pressure_at_frame': float(pressure_at_frame) if pressure_at_frame is not None else -1.0,
        'elapsed_time': float(elapsed_time) if elapsed_time is not None else -1.0,
        'schedule_time': float(schedule_time) if schedule_time is not None else -1.0,
        'crossing_time': float(crossing_time),
        'crossing_fill': float(crossing_fill),
        'crossing_alpha': float(crossing_alpha),
        'pressure_at_crossing': float(pressure_at_crossing) if pressure_at_crossing is not None else -1.0,
        'status': status,
        'is_eligible': bool(is_eligible)
    }



def make_history_signature(fill, top_area, min_thickness):
    """
    Compact numerical signature of one frame geometry.

    It is used only to detect frames copied from the restart source ODB.
    We intentionally do not include pressure, because copied geometric frames
    are identical while the tested pressure value is different.
    """
    tol = float(INHERITED_FRAME_MATCH_TOL)
    if tol <= 0.0:
        tol = 1.0e-7

    return (
        int(round(float(fill) / tol)),
        int(round(float(top_area) / tol)),
        int(round(float(min_thickness) / tol))
    )


def collect_inherited_history_signatures(restart_source):
    """
    Read the source restart ODB and collect signatures of frames that are later
    than the selected restart increment.

    Why this is needed:
        Abaqus restart ODBs can contain copied history frames from the source
        analysis. If these frames are scanned in a new candidate job, all
        candidate pressures may be evaluated at exactly the same old frame.
        This function lets evaluate_job_at_target_fill skip such frames.
    """
    if (not SKIP_INHERITED_RESTART_HISTORY) or restart_source is None:
        return set()

    source_job = restart_source.get('job_name', None)
    if source_job is None:
        return set()

    source_odb_path = get_odb_path(source_job)
    if not os.path.exists(source_odb_path):
        print('WARNING: source restart ODB not found for inherited history filtering: %s' % source_odb_path)
        return set()

    restart_inc = int(restart_source.get('increment_number', -1))
    signatures = set()
    odb = None

    try:
        odb = visualization.openOdb(source_odb_path)

        if STEP_NAME not in odb.steps:
            odb.close()
            return set()

        step = odb.steps[STEP_NAME]
        instance = find_instance(odb)
        if instance is None:
            odb.close()
            return set()

        for frame in step.frames:
            try:
                if int(frame.incrementNumber) <= restart_inc:
                    continue

                top_points, bottom_points, top_labels, bottom_labels = extract_top_bottom_layers(frame, instance)
                fill, top_area = compute_fill(top_points)
                min_data = min_distance_to_top(
                    bottom_points,
                    top_points,
                    bottom_labels=bottom_labels,
                    n=THICKNESS_SEARCH_WINDOW
                )
                signatures.add(make_history_signature(fill, top_area, min_data[0]))

            except Exception:
                continue

        odb.close()
        print('Inherited history filter: collected %d source-frame signatures from %s after increment %d.' %
              (len(signatures), source_job, restart_inc))
        return signatures

    except Exception:
        print('WARNING: failed to collect inherited history signatures from source ODB: %s' % source_odb_path)
        traceback.print_exc()
        try:
            if odb is not None:
                odb.close()
        except Exception:
            pass
        return set()



def compute_target_crossing(prev_item, curr_frame_value, curr_fill, target_fill):
    """
    Estimate the time at which the target filling was crossed.

    The restart increment is still the current saved frame, because Abaqus can
    restart only from a written increment. This function only improves the
    exported pressure-time schedule and diagnostics.

    If the previous processed frame is unavailable, or if interpolation is not
    possible, the current frame time is returned.
    """
    curr_time = float(curr_frame_value)
    curr_fill = float(curr_fill)
    target = float(target_fill)

    if (not USE_INTERPOLATED_TARGET_TIME) or prev_item is None:
        return curr_time, curr_fill, 1.0

    prev_fill = float(prev_item.get('fill', curr_fill))
    prev_time = float(prev_item.get('frame_value', curr_time))

    if prev_fill >= target:
        return curr_time, curr_fill, 1.0

    denom = curr_fill - prev_fill
    if abs(denom) <= 1.0e-14:
        return curr_time, curr_fill, 1.0

    alpha = (target - prev_fill) / denom
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    crossing_time = prev_time + alpha * (curr_time - prev_time)
    return float(crossing_time), float(target), float(alpha)

def evaluate_job_at_target_fill(job_name, target_fill, stage=None, pressure=None, restart_source=None):
    odb_path = get_odb_path(job_name)
    if not os.path.exists(odb_path):
        print('ODB not found: %s' % odb_path)
        return None

    odb = None
    try:
        odb = visualization.openOdb(odb_path)

        if STEP_NAME not in odb.steps:
            print('Step not found in ODB: %s' % STEP_NAME)
            odb.close()
            return None

        step = odb.steps[STEP_NAME]
        instance = find_instance(odb)
        if instance is None:
            odb.close()
            return None

        max_reached_fill = -1.0
        best_reached = None
        best_closest = None

        stage_start_time = get_stage_start_time(restart_source)
        stage_start_pressure = get_stage_start_pressure(restart_source)
        stage_schedule_start_time = get_schedule_start_time(restart_source)

        inherited_signatures = collect_inherited_history_signatures(restart_source)
        skipped_by_inherited_history = 0
        skipped_by_elapsed_time = 0
        processed_current_stage_frames = 0
        prev_processed_item = None

        use_inherited_history_filter = False
        if restart_source is not None and inherited_signatures:
            if SKIP_INHERITED_ONLY_IF_PRESSURE_CHANGES:
                # If the candidate pressure is equal to the pressure already active
                # in the selected restart frame, the source-job tail after the
                # restart increment is a valid same-pressure continuation.
                # Skipping it causes artificial jumps, e.g. selecting F=0.41
                # when the real target F=0.30 was crossed earlier in that tail.
                use_inherited_history_filter = (abs(float(pressure) - float(stage_start_pressure)) > float(PRESSURE_EQUAL_TOL))
            else:
                use_inherited_history_filter = True

        print('Inherited-history skip mode for job %s: use_filter=%s, candidate_pressure=%.8f, start_pressure=%.8f' % (
            job_name, str(bool(use_inherited_history_filter)), float(pressure), float(stage_start_pressure)))

        for idx, frame in enumerate(step.frames):
            try:
                top_points, bottom_points, top_labels, bottom_labels = extract_top_bottom_layers(frame, instance)
                fill, top_area = compute_fill(top_points)

                min_data = min_distance_to_top(
                    bottom_points,
                    top_points,
                    bottom_labels=bottom_labels,
                    n=THICKNESS_SEARCH_WINDOW
                )

                min_thickness_for_log = min_data[0]

            except Exception:
                print('WARNING: failed to process frame %d in job %s' % (idx, job_name))
                traceback.print_exc()
                continue

            elapsed_time = compute_frame_elapsed_time(restart_source, float(frame.frameValue))
            schedule_time = stage_schedule_start_time + float(elapsed_time)

            current_pressure_at_frame = pressure_value_after_elapsed(
                stage_start_pressure,
                float(pressure),
                elapsed_time
            )

            # Do not evaluate copied source-history frames in restart jobs.
            # These frames can be identical for all candidate pressures and
            # therefore lead to identical fill/min_thickness on a stage.
            if restart_source is not None:
                if elapsed_time <= float(MIN_CURRENT_STAGE_ELAPSED_TIME):
                    skipped_by_elapsed_time += 1
                    continue

                if use_inherited_history_filter:
                    sig = make_history_signature(fill, top_area, min_data[0])
                    if sig in inherited_signatures:
                        skipped_by_inherited_history += 1
                        continue

            processed_current_stage_frames += 1

            if fill > max_reached_fill:
                max_reached_fill = fill

            append_live_increment_log(
                stage=int(stage),
                pressure=float(pressure),
                job_name=job_name,
                frame_index=int(idx),
                increment_number=int(frame.incrementNumber),
                frame_value=float(frame.frameValue),
                top_area=float(top_area),
                target_fill=float(target_fill),
                fill=float(fill),
                min_thickness=float(min_thickness_for_log)
            )

            crossing_time = float(frame.frameValue)
            crossing_fill = float(fill)
            crossing_alpha = 1.0
            pressure_at_crossing = current_pressure_at_frame

            if fill >= target_fill:
                crossing_time, crossing_fill, crossing_alpha = compute_target_crossing(
                    prev_processed_item,
                    float(frame.frameValue),
                    float(fill),
                    float(target_fill)
                )
                pressure_at_crossing = pressure_value_at_time(
                    stage_start_time,
                    stage_start_pressure,
                    float(pressure),
                    crossing_time
                )

            temp_item = make_result_item(
                job_name=job_name,
                idx=idx,
                frame=frame,
                target_fill=target_fill,
                fill=fill,
                top_area=top_area,
                max_reached_fill=max_reached_fill,
                min_data=min_data,
                status=STATUS_FAILED,
                is_eligible=False,
                pressure_at_frame=current_pressure_at_frame,
                elapsed_time=elapsed_time,
                schedule_time=schedule_time,
                crossing_time=crossing_time,
                crossing_fill=crossing_fill,
                crossing_alpha=crossing_alpha,
                pressure_at_crossing=pressure_at_crossing
            )

            if best_closest is None or temp_item['fill_error'] < best_closest['fill_error']:
                best_closest = temp_item

            if fill >= target_fill:
                # Use the first chronological crossing after filtering.
                # This is the frame from which the stage should be considered
                # to have reached the target. Choosing a later frame with a
                # smaller overshoot can break replay consistency.
                if best_reached is None:
                    best_reached = temp_item

            prev_processed_item = temp_item

        print('Frame filter for job %s: processed_current_stage_frames=%d, skipped_inherited=%d, skipped_elapsed=%d, inherited_signatures=%d, use_inherited_filter=%s' % (
            job_name,
            int(processed_current_stage_frames),
            int(skipped_by_inherited_history),
            int(skipped_by_elapsed_time),
            int(len(inherited_signatures)),
            str(bool(use_inherited_history_filter))
        ))

        odb.close()

        if best_reached is not None:
            if best_reached['fill_error'] <= FILL_TOLERANCE:
                status = STATUS_OK
                eligible = True
            else:
                status = STATUS_REACHED_LARGE_ERROR
                eligible = not STRICT_FILL_TOLERANCE

            best_reached['status'] = status
            best_reached['is_eligible'] = eligible
            best_reached['max_reached_fill'] = max_reached_fill
            return best_reached

        if best_closest is not None:
            best_closest['status'] = STATUS_NOT_REACHED
            best_closest['is_eligible'] = False
            best_closest['max_reached_fill'] = max_reached_fill
            return best_closest

        return None

    except Exception:
        print('ODB postprocessing failed for job: %s' % job_name)
        traceback.print_exc()
        try:
            if odb is not None:
                odb.close()
        except Exception:
            pass
        return None


# ============================================================
# MAIN OPTIMIZATION LOOP
# ============================================================

def choose_best_candidate(candidates):
    """
    Select the best eligible candidate.

    Main criterion:
        maximum minimum thickness.

    Tie rule requested for the final version:
        if several pressures give the same thinning when min_thickness is rounded
        to the third decimal place, choose the smallest pressure.

    Additional tie-breaker:
        if pressure is also the same, choose the candidate closer to target filling.
    """
    eligible = []
    for item in candidates:
        if item['result']['is_eligible']:
            eligible.append(item)

    if not eligible:
        return None

    eligible_sorted = sorted(
        eligible,
        key=lambda item: (
            round(float(item['result']['min_thickness']), 2),
            -float(item['pressure']),
            -float(item['result']['fill_error'])
        ),
        reverse=True
    )
    return eligible_sorted[0]


def pressure_optimization_by_fill():
    if MODEL_NAME not in mdb.models:
        print('ERROR: model not found: %s' % MODEL_NAME)
        return

    init_logs()
    ensure_restart_output_keyword(MODEL_NAME)

    print('=' * 100)
    print('START FILL-BASED PRESSURE OPTIMIZATION')
    print('=' * 100)

    restart_source = None
    current_fill = 0.0
    final_best = None
    final_pressure_schedule = []      # exported continuous replay schedule; uses real restart-frame times
    restart_pressure_history = []     # exact history used in restart candidate amplitudes
    interpolated_pressure_schedule = []  # diagnostic only; not recommended for final replay

    fill_targets = make_fill_targets(FILL_STEP, FINAL_FILL_TARGET)
    print('Fill targets: %s' % ', '.join(['%.6f' % v for v in fill_targets]))

    for stage, target_fill in enumerate(fill_targets, start=1):

        print('\n' + '=' * 100)
        print('STAGE %d: target filling %.6f -> %.6f' % (stage, current_fill, target_fill))
        if restart_source is None:
            print('Stage start: initial model, absolute step time = 0.000000')
        else:
            print('Stage start: restart from job=%s, inc=%d, frame_time=%.6f, fill=%.6f, pressure_at_frame=%.6f' % (
                restart_source['job_name'],
                int(restart_source['increment_number']),
                float(restart_source.get('frame_value', 0.0)),
                float(restart_source.get('fill', -1.0)),
                float(restart_source.get('pressure_at_frame', -1.0))
            ))
        print('=' * 100)

        candidates = []

        for pressure in PRESSURE_VALUES:
            job_name = '%s_s%03d_p%s' % (BASE_JOB_PREFIX, stage, pressure_to_tag(pressure))

            ok = run_job(job_name, pressure, restart_source)
            if not ok:
                continue

            result = evaluate_job_at_target_fill(job_name, target_fill, stage, pressure, restart_source)
            if result is None:
                print('No valid result for job: %s' % job_name)
                continue

            candidates.append({
                'pressure': float(pressure),
                'job_name': job_name,
                'result': result
            })

            print('Candidate: p=%.6f, status=%s, eligible=%s, fill=%.6f, max_fill=%.6f, min_t=%.6f, frame=%d, inc=%d' %
                  (pressure,
                   result['status'],
                   str(bool(result['is_eligible'])),
                   result['fill'],
                   result['max_reached_fill'],
                   result['min_thickness'],
                   result['frame_index'],
                   result['increment_number']))

        best = choose_best_candidate(candidates)

        for cand in candidates:
            is_best = False
            if best is not None and cand['job_name'] == best['job_name']:
                is_best = True
            append_candidate_to_logs(stage, cand['pressure'], cand['result'], is_best)

        if best is None:
            print('\nSTOP: no eligible candidates at stage %d.' % stage)
            print('Possible reasons:')
            print('  1. STAGE_TIME is too short;')
            print('  2. pressures are too low;')
            print('  3. filling calculation does not match the physical filling;')
            print('  4. TOP/BOTTOM layer extraction settings are wrong.')
            print('Check %s and %s.' % (LIVE_INCREMENT_LOG, SUMMARY_CSV))
            break

        best_res = best['result']
        final_best = best

        print('\nBEST AT STAGE %d:' % stage)
        print('  pressure      = %.6f' % best['pressure'])
        print('  job           = %s' % best['job_name'])
        print('  odb           = %s' % best_res['odb_name'])
        print('  status        = %s' % best_res['status'])
        print('  target fill   = %.6f' % best_res['target_fill'])
        print('  actual fill   = %.6f' % best_res['fill'])
        print('  max fill      = %.6f' % best_res['max_reached_fill'])
        print('  top area      = %.6f' % best_res['top_area'])
        print('  min thickness = %.6f' % best_res['min_thickness'])
        print('  restart inc   = %d' % best_res['increment_number'])

        append_best_path(stage, best['pressure'], best_res)
        write_final_result(best_res, best['pressure'])

        # Exact restart history: switch/end times are real saved ODB frame times.
        # This is the history used when creating the next restart candidate.
        extend_pressure_schedule(
            restart_pressure_history,
            best['pressure'],
            best_res,
            restart_source,
            use_interpolated_time=False
        )

        # Exported continuous replay schedule:
        # use the same real saved-frame times as the restart chain.
        # Do NOT switch at crossing_time here: Abaqus did not actually restart
        # from the interpolated state, so this would create a different loading path.
        extend_pressure_schedule(
            final_pressure_schedule,
            best['pressure'],
            best_res,
            restart_source,
            use_interpolated_time=False
        )

        # Diagnostic schedule only: interpolated target-crossing times.
        # This file is useful for checking target crossing times, but should not
        # be used as the main pressure amplitude for the verification run.
        extend_pressure_schedule(
            interpolated_pressure_schedule,
            best['pressure'],
            best_res,
            restart_source,
            use_interpolated_time=True
        )

        write_pressure_schedule_csv(final_pressure_schedule, PRESSURE_SCHEDULE_CSV)
        if WRITE_RESTART_PRESSURE_HISTORY_CSV:
            write_pressure_schedule_csv(restart_pressure_history, RESTART_PRESSURE_SCHEDULE_CSV)
            write_pressure_schedule_csv(interpolated_pressure_schedule,
                                        os.path.join(LOG_DIR, 'fill_pressure_optimizer_pressure_schedule_interpolated_diagnostic.csv'))

        restart_source = {
            'job_name': best['job_name'],
            'step_name': STEP_NAME,
            'increment_number': best_res['increment_number'],
            'frame_value': best_res['frame_value'],
            'fill': best_res['fill'],
            'selected_pressure': best['pressure'],
            'pressure_at_frame': best_res['pressure_at_frame'],
            'elapsed_time': best_res.get('elapsed_time', -1.0),
            'schedule_time': best_res.get('schedule_time', best_res['frame_value']),
            'pressure_history': restart_pressure_history[:],
            'export_pressure_schedule': final_pressure_schedule[:],
            'interpolated_pressure_schedule': interpolated_pressure_schedule[:],
            'crossing_time': best_res.get('crossing_time', best_res['frame_value'])
        }

        current_fill = target_fill

        if abs(current_fill - FINAL_FILL_TARGET) <= 1.0e-8:
            print('\nTarget FINAL_FILL_TARGET reached.')

    print('\nPROCESS FINISHED')
    print('Log file: %s' % LOG_FILE)
    print('Summary CSV: %s' % SUMMARY_CSV)
    print('Live increment log: %s' % LIVE_INCREMENT_LOG)
    print('Best path CSV: %s' % BEST_PATH_CSV)
    print('Pressure schedule CSV: %s' % PRESSURE_SCHEDULE_CSV)
    if WRITE_RESTART_PRESSURE_HISTORY_CSV:
        print('Restart pressure history CSV: %s' % RESTART_PRESSURE_SCHEDULE_CSV)
    print('Final result TXT: %s' % FINAL_RESULT_TXT)

    if final_best is not None:
        print('Final selected ODB: %s' % final_best['result']['odb_name'])


def main():
    try:
        pressure_optimization_by_fill()
    finally:
        clear_restart_settings()


if __name__ == '__main__':
    main()
