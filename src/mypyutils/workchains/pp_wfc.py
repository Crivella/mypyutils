from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.engine import WorkChain, ToContext

from aiida_quantumespresso.utils.mapping import prepare_process_inputs

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PpCalculation   = CalculationFactory('quantumespresso.pp')

class PPwfcWorkChain(WorkChain):
    """Workchain to calculate the wavefunction at a given set of kpoints."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'})
        spec.expose_inputs(PwBaseWorkChain, namespace='nscf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for the NSCF calculation.'})
        spec.expose_inputs(PpCalculation, namespace='pp',
            exclude=('parent_folder', ),
            namespace_options={'help': 'Inputs for the `PpCalculation` for the PP calculation.'})
        spec.input('structure', valid_type=orm.StructureData, help='The inputs structure.')
        spec.input('kpoints', valid_type=orm.KpointsData,
            help='Explicit kpoints to use for the NSCF calculation.')
        spec.input('wavefunction_min', valid_type=orm.Int, required=False,
            help='Smallest index of wavefunctions to compute.')
        spec.input('wavefunction_max', valid_type=orm.Int, required=False,
            help='Largest index of wavefunctions to compute.')
        spec.input('wavefunction_ef_min', valid_type=orm.Int, required=False,
            help='Smallest index, with respoect to the fermi level, of wavefunctions to compute.')
        spec.input('wavefunction_ef_max', valid_type=orm.Int, required=False,
            help='Largest index, with respoect to the fermi level, of wavefunctions to compute.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        # spec.inputs.validator = validate_inputs
        spec.outline(
            cls.setup,
            cls.run_scf,
            cls.inspect_scf,
            cls.run_nscf,
            cls.inspect_nscf,
            cls.run_pp,
            cls.inspect_pp,
            cls.results,
        )

        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='The scf PwBasexWorkChain sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_NSCF',
            message='The nscf PwBasexWorkChain sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_PP',
            message='The PpCalculation sub process failed')

        spec.output('scf_parameters', valid_type=orm.Dict,
            help='The output parameters of the SCF `PwBaseWorkChain`.')
        spec.output('nscf_parameters', valid_type=orm.Dict,
            help='The output parameters of the NSCF `PwBaseWorkChain`.')
        spec.output('pp_parameters', valid_type=orm.Dict,
            help='The output parameters of the `PpCalculation`.')
        spec.output('pp_retrieved', valid_type=orm.FolderData,
            help='The retrieved folder of the `PpCalculation`.')
        spec.output('pp_data', valid_type=orm.ArrayData,
            help='The output_data node of the `PpCalculation`.')
        # yapf: enable


    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure
        self.ctx.bands_kpoints = self.inputs.get('kpoints', None)

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.metadata.call_link_label = 'scf'
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})['calculation'] = 'scf'

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launching PwBaseWorkChain<{running.pk}> in scf mode')

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(f'scf PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.ctx.current_number_of_bands = workchain.outputs.output_parameters.get_attribute('number_of_bands')

    def run_nscf(self):
        """Run the PwBaseWorkChain in bands mode along the path of high-symmetry determined by seekpath."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='bands'))
        inputs.metadata.call_link_label = 'bands'
        inputs.kpoints = self.ctx.bands_kpoints
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.current_folder
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters.setdefault('SYSTEM', {})
        inputs.pw.parameters.setdefault('ELECTRONS', {})

        # The following flags always have to be set in the parameters, regardless of what caller specified in the inputs
        inputs.pw.parameters['CONTROL']['calculation'] = 'bands'

        # Only set the following parameters if not directly explicitly defined in the inputs
        inputs.pw.parameters['ELECTRONS'].setdefault('diagonalization', 'cg')
        inputs.pw.parameters['ELECTRONS'].setdefault('diago_full_acc', True)

        # If `nbands_factor` is defined in the inputs we set the `nbnd` parameter
        factor = 1.2

        parameters = self.ctx.workchain_scf.outputs.output_parameters.get_dict()
        if int(parameters['number_of_spin_components']) > 1:
            nspin_factor = 2
        else:
            nspin_factor = 1
        nbands = int(parameters['number_of_bands'])
        nelectron = int(parameters['number_of_electrons'])
        nbnd = max(
            int(0.5 * nelectron * nspin_factor * factor),
            int(0.5 * nelectron * nspin_factor) + 4 * nspin_factor, nbands
        )
        inputs.pw.parameters['SYSTEM']['nbnd'] = nbnd

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launching PwBaseWorkChain<{running.pk}> in nscf mode')

        return ToContext(workchain_nscf=running)

    def inspect_nscf(self):
        """Verify that the PwBaseWorkChain for the bands run finished successfully."""
        workchain = self.ctx.workchain_nscf

        if not workchain.is_finished_ok:
            self.report(f'nscf PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.current_folder = workchain.outputs.remote_folder

    def run_pp(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(self.exposed_inputs(PpCalculation, namespace='pp'))
        inputs.metadata.call_link_label = 'pp'
        inputs.parent_folder = self.ctx.current_folder

        nscf_params = self.ctx.workchain_nscf.outputs.output_parameters

        inputs.parameters.setdefault('INPUTPP', {})
        inputs.parameters.setdefault('PLOT', {})
        inputs.parameters['INPUTPP']['plot_num'] = 7
        nel = int(nscf_params['number_of_electrons'])
        if 'wavefunction_min' in self.inputs:
            kband_min = self.inputs.wavefunction_min.value
        else:
            if 'wavefunction_ef_min' in self.inputs:
                i = self.inputs.wavefunction_ef_min.value - 1
                kband_min = nel - i
            else:
                kband_min = 1

        if 'wavefunction_max' in self.inputs:
            kband_max = self.inputs.wavefunction_max.value
        else:
            if 'wavefunction_ef_max' in self.inputs:
                i = self.inputs.wavefunction_ef_max.value 
                kband_max = nel + i
            else:
                kband_max = nscf_params['number_of_atomic_wfc']

        inputs.parameters['INPUTPP']['kband(1)'] = kband_min
        inputs.parameters['INPUTPP']['kband(2)'] = kband_max
        inputs.parameters['INPUTPP']['kpoint(1)'] = 1
        inputs.parameters['INPUTPP']['kpoint(2)'] = nscf_params['number_of_k_points']

        inputs.parameters['PLOT']['iflag'] = 3
        inputs.parameters['PLOT']['output_format'] = 5

        inputs = prepare_process_inputs(PpCalculation, inputs)
        running = self.submit(PpCalculation, **inputs)

        self.report(f'launching PpCalculation<{running.pk}>')

        return ToContext(workchain_pp=running)

    def inspect_pp(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_pp

        if not workchain.is_finished_ok:
            self.report(f'PpCalculation failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PP

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain succesfully completed')
        self.out('scf_parameters', self.ctx.workchain_scf.outputs.output_parameters)
        self.out('nscf_parameters', self.ctx.workchain_nscf.outputs.output_parameters)
        self.out('pp_parameters', self.ctx.workchain_pp.outputs.output_parameter)
        self.out('pp_retrieved', self.ctx.workchain_pp.outputs.retrieved)
        self.out('pp_data', self.ctx.workchain_pp.outputs.output_data)

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
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

