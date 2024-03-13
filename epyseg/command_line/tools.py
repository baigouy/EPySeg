import shlex

def join_command(command):
    command_string = shlex.join(command)
    return command_string

if __name__ == '__main__':
    command = [
        '/home/aigouy/UCSC_tools/blastn',
        '-task', 'blastn',
        '-query', '/tmp/tmp4a97pceg.fasta',
        '-db', '/F/blast_outputs/25kb/D_suzukii_real_25kb_for_blast.fasta',
        '-evalue', '50',
        '-num_threads', '4',
        '-outfmt', '5',
        '-out', '/F/blast_outputs/subsets/D_suzukii_real_25kb_for_blast_F1a-b.xml'
    ]
    print(join_command(command))
