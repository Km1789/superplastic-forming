from abaqus import session
from odbAccess import openOdb
from abaqusConstants import NODAL, INTEGRATION_POINT  # Added import
import os


def export_rpt_files(odb_name, output_dir, increment_step):
    """
    Exports .rpt files for each increment with a given step.
    :param odb_name: Name of the .odb file
    :param output_dir: Directory to save .rpt files
    :param increment_step: Increment step for export
    """

    # Open the ODB file
    odb = openOdb(odb_name)
    step_name = odb.steps.keys()[0]  # Take the first step name as a string
    step = odb.steps[step_name]
    step_index = odb.steps.keys().index(step_name)  # Get step index

    # Create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine the number of increments
    num_frames = len(step.frames)

    # Iterate through increments with the given step
    for i in range(0, num_frames, increment_step):
        frame = step.frames[i]
        frame_time = frame.frameValue  # Get the time corresponding to this increment
        rpt_filename = os.path.join(output_dir, 'increment_{}.rpt'.format(i))

        # Open report file
        session.viewports['Viewport: 1'].setValues(displayedObject=odb)
        session.fieldReportOptions.setValues(printTotal=True)

        # Write time information to the report file
        with open(rpt_filename, 'w') as rpt_file:
            rpt_file.write('Time: {:.6f}\n'.format(frame_time))

        # Export report
        session.writeFieldReport(
            fileName=rpt_filename,
            append=ON,  # Append data after writing time
            sortItem='Node Label',
            odb=odb,
            step=step_index,  # Pass step index instead of step name
            frame=i,
            outputPosition=NODAL,
            variable=(
                ('COORD', NODAL),
            )
        )
        print('Saved file:', rpt_filename, 'with time:', frame_time)

    odb.close()
    print('Export completed.')


# Example usage
# export_rpt_files('algo4.odb', r'data/forming/Free_bulg_test/Reports/Smirnov/Frictionless/algo4', 1)
# export_rpt_files('pressure_1_25.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_1_25', 2)
# export_rpt_files('pressure_1_0.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_1_0', 2)
# export_rpt_files('pressure_1_5.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_1_5', 2)
# export_rpt_files('pressure_2_0.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_2_0', 2)
# export_rpt_files('pressure_2_5.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_2_5', 2)
# export_rpt_files('pressure_3_0.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_3_0', 2)
# export_rpt_files('pressure_3_5.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_3_5', 2)
# export_rpt_files('pressure_0_5.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_0_5', 5)
# export_rpt_files('pressure_0_75.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_0_75', 5)
# export_rpt_files('pressure_1_75.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_1_75', 2)
# export_rpt_files('pressure_2_25.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_2_25', 2)
# export_rpt_files('pressure_2_75.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_2_75', 2)
# export_rpt_files('pressure_3_25.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_3_25', 2)
# export_rpt_files('pressure_3_75.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_3_75', 2)
# export_rpt_files('pressure_4_0.odb', r'data/forming/Free_bulg_test/Reports/Backofen/Frictionless/pressure_4_0', 2)
# export_rpt_files('alg_try.odb', r'data/forming/Free_bulg_test/Reports/Smirnov/Frictionless/algorithm_try_0_8', 1)
# export_rpt_files('sr_opt_center.odb', r'data/forming/Free_bulg_test/Reports/Smirnov/Frictionless/sr_opt_center', 2)
export_rpt_files('smoth_alg.odb', r'data/forming/Free_bulg_test/Reports/Smirnov/Frictionless/smooth_alg', 2)