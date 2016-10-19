
import os
import os.path

def merge_logs(logs_dir):

    written = False

    # All direct sub directories of the logs directory
    dirs = [os.path.join(logs_dir, o) for o in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, o))]

    # Make sure directories are sorted otherwise Rickshaw will complain
    dirs.sort()

    # Let's first populate a temporal file and rename it afterwards - guarantees an interruption free web experience
    nhf = '%s/%s' % (logs_dir, 'new_hyper.js')

    with open(nhf, 'w') as dump_file:
        # Assigning a global variable that the report page can pick up after loading the data as a regular script
        dump_file.write('window.ALL_THE_DATA = [')
        for d in dirs:
            hf = os.path.join(d, "hyper.json")
            if os.path.isfile(hf):
                # Separate by comma if there was already something written
                if written:
                    dump_file.write(',\n')
                written = True
                # Append the whole file
                dump_file.write(open(hf, 'r').read())
        dump_file.write('];')

    # Finally we rename the temporal file and overwrite a potentially existing active one
    os.rename(nhf, '%s/%s' % (logs_dir, 'hyper.js'))
