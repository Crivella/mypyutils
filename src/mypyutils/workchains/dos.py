# -*- coding: utf-8 -*-
"""Workchain to compute a band structure for a given structure using Quantum ESPRESSO pw.x."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.engine import WorkChain, ToContext, if_

from aiida_quantumespresso.utils.mapping import prepare_process_inputs

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
DosCalc = CalculationFactory('quantumespresso.dos')

class DosWorkChain(WorkChain):
    """Workchain to compute a DOS for a given structure using Quantum ESPRESSO pw.x. """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        # spec.expose_inputs(PwRelaxWorkChain, namespace='relax', exclude=('clean_workdir', 'structure'),
        #     namespace_options={'required': False, 'populate_defaults': False,
        #     'help': 'Inputs for the `PwRelaxWorkChain`, if not specified at all, the relaxation step is skipped.'})
        spec.expose_inputs(PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'})
        spec.expose_inputs(PwBaseWorkChain, namespace='nscf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for the NSCF calculation.'})
        spec.expose_inputs(DosCalc, namespace='dos',
            exclude=('parent_folder', ),
            namespace_options={'help': 'Inputs for the `DosCalculation` for the DOS calculation.'})
        spec.input('parent_folder', valid_type=orm.RemoteData, required=False)
        spec.input('structure', valid_type=orm.StructureData, help='The inputs structure.')
        # spec.input('kpoints_distance', valid_type=orm.Float, required=False,
        #     help='The minimum desired distance in 1/Å between k-points in reciprocal space. The explicit k-points will '
        #          'be generated automatically by a calculation function based on the input structure.')
        spec.input('nbands_factor', valid_type=orm.Float, default=lambda: orm.Float(1.5),
            help='The number of bands for the BANDS calculation is that used for the SCF multiplied by this factor.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.outline(
            cls.setup,
            if_(cls.should_do_scf)(
                cls.run_scf,
                cls.inspect_scf,
                ),
            cls.run_nscf,
            cls.inspect_nscf,
            cls.run_dos,
            cls.inspect_dos,
            cls.results,
        )
        spec.exit_code(201, 'ERROR_INVALID_INPUT_NUMBER_OF_BANDS',
            message='Cannot specify both `nbands_factor` and `bands.pw.parameters.SYSTEM.nbnd`.')
        spec.exit_code(202, 'ERROR_INVALID_INPUT_KPOINTS',
            message='Cannot specify both `bands_kpoints` and `bands_kpoints_distance`.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='The PwRelaxWorkChain sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='The scf PwBasexWorkChain sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_NSCF',
            message='The bands PwBasexWorkChain sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_DOS',
            message='The dos DosCalculation sub process failed')

        spec.output('scf_remote_folder', valid_type=orm.RemoteData)
        spec.output('nscf_remote_folder', valid_type=orm.RemoteData)
        spec.output('scf_parameters', valid_type=orm.Dict,
            help='The output parameters of the SCF `PwBaseWorkChain`.')
        spec.output('nscf_parameters', valid_type=orm.Dict,
            help='The output parameters of the NSCF `PwBaseWorkChain`.')
        spec.output('dos_parameters', valid_type=orm.Dict,
            help='The output parameters of the DOS calculation.')
        spec.output('output_dos', valid_type=orm.XyData)


    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure
        self.ctx.current_number_of_bands = None
        # self.ctx.bands_kpoints = self.inputs.get('bands_kpoints', None)

    def should_do_scf(self):
        if 'parent_folder' in self.inputs:
            remote = self.inputs.parent_folder
            self.ctx.current_folder = remote
            self.ctx.workchain_scf = remote.creator
            self.ctx.current_number_of_bands = remote.creator.outputs.output_parameters.get_attribute('number_of_bands')
            return False

        return True

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.metadata.call_link_label = 'scf'
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})['calculation'] = 'scf'

        # Make sure to carry the number of bands from the relax workchain if it was run and it wasn't explicitly defined
        # in the inputs. One of the base workchains in the relax workchain may have changed the number automatically in
        #  the sanity checks on band occupations.
        if self.ctx.current_number_of_bands:
            inputs.pw.parameters.setdefault('SYSTEM', {}).setdefault('nbnd', self.ctx.current_number_of_bands)

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(running.pk, 'scf'))

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report('scf PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.ctx.current_number_of_bands = workchain.outputs.output_parameters.get_attribute('number_of_bands')

        self.out('scf_remote_folder', workchain.outputs.remote_folder)
        self.out('scf_parameters', workchain.outputs.output_parameters)

    def run_nscf(self):
        """Run the PwBaseWorkChain in nscf mode along the path of high-symmetry determined by seekpath."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='nscf'))
        inputs.metadata.call_link_label = 'nscf'
        # inputs.pw.metadata.options.max_wallclock_seconds *= 4
        # inputs.kpoints_distance = self.inputs.kpoints_distance
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.current_folder
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters.setdefault('SYSTEM', {})
        inputs.pw.parameters.setdefault('ELECTRONS', {})

        # The following flags always have to be set in the parameters, regardless of what caller specified in the inputs
        inputs.pw.parameters['CONTROL']['calculation'] = 'nscf'

        # Only set the following parameters if not directly explicitly defined in the inputs
        # inputs.pw.parameters['ELECTRONS'].setdefault('diagonalization', 'cg')
        # inputs.pw.parameters['ELECTRONS'].setdefault('diago_full_acc', True)

        # If `nbands_factor` is defined in the inputs we set the `nbnd` parameter
        if 'nbands_factor' in self.inputs:
            factor = self.inputs.nbands_factor.value
            parameters = self.ctx.workchain_scf.outputs.output_parameters.get_dict()
            nspin = int(parameters['number_of_spin_components'])
            nbands = int(parameters['number_of_bands'])
            nelectron = int(parameters['number_of_electrons'])
            nbnd = max(int(0.5 * nelectron * nspin * factor), int(0.5 * nelectron * nspin) + 4 * nspin, nbands)
            inputs.pw.parameters['SYSTEM']['nbnd'] = nbnd

        # Otherwise set the current number of bands, unless explicitly set in the inputs
        else:
            inputs.pw.parameters['SYSTEM'].setdefault('nbnd', self.ctx.current_number_of_bands)

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(running.pk, 'nscf'))

        return ToContext(workchain_nscf=running)


    def inspect_nscf(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_nscf

        if not workchain.is_finished_ok:
            self.report('scf PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.ctx.current_number_of_bands = workchain.outputs.output_parameters.get_attribute('number_of_bands')

        self.out('nscf_remote_folder', workchain.outputs.remote_folder)
        self.out('nscf_parameters', workchain.outputs.output_parameters)

    def run_dos(self):
        """Run the PwBaseWorkChain in bands mode along the path of high-symmetry determined by seekpath."""
        inputs = AttributeDict(self.exposed_inputs(DosCalc, namespace='dos'))
        inputs.metadata.call_link_label = 'dos'
        inputs.parent_folder = self.ctx.current_folder

        inputs = prepare_process_inputs(DosCalc, inputs)
        running = self.submit(DosCalc, **inputs)

        self.report('launching DosCalculation<{}> in {} mode'.format(running.pk, 'dos'))

        return ToContext(workchain_dos=running)

    def inspect_dos(self):
        """Verify that the PwBaseWorkChain for the bands run finished successfully."""
        workchain = self.ctx.workchain_dos

        if not workchain.is_finished_ok:
            self.report('DOS failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.out('dos_parameters', workchain.outputs.output_parameters)
        self.out('output_dos', workchain.outputs.output_dos)

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain succesfully completed')
        # self.out('scf_parameters', self.ctx.workchain_scf.outputs.output_parameters)
        # self.out('band_parameters', self.ctx.workchain_bands.outputs.output_parameters)
        # self.out('band_structure', self.ctx.workchain_bands.outputs.output_band)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report('cleaned remote folders of calculations: {}'.format(' '.join(map(str, cleaned_calcs))))