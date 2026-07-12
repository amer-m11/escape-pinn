from ..configuration import (
    FCMLPConfiguration,
    NetworkConfiguration,
    PhysicalSize,
    SIRENConfiguration,
)
from .base_network import EscapeNet
from .fc_mlp_network import FCMLPNet2D, FCMLPNet3D, FCMLPNetCartesian3D
from .siren_network import SIRENNet2D, SIRENNet3D, SIRENNetCartesian3D


class NetworkFactory:
    """Factory for constructing PINN network architectures from configuration.

    Dispatches on network type, dim, parametrization.
    """

    @staticmethod
    def create_network(
        config: NetworkConfiguration,
        physical_size: PhysicalSize,
        parametrization: str = "spherical",
    ) -> EscapeNet:
        size = physical_size.as_tuple()
        dim = physical_size.dim
        cartesian = dim == 3 and parametrization == "cartesian"

        if isinstance(config, FCMLPConfiguration):
            ff = config.fourier_features
            kwargs = dict(
                hidden_layers=config.hidden_layers,
                hidden_neurons=config.hidden_neurons,
                arch_name=config.arch_name,
                fourier_n_features=ff.n_features if ff is not None else None,
                fourier_sigma=ff.sigma if ff is not None else None,
                trunk=config.trunk,
                unit_norm_output=config.unit_norm_output,
                factored_eikonal=config.factored_eikonal,
                per_channel_heads=config.per_channel_heads,
            )
            if config.selector_head:
                if dim != 3:
                    raise NotImplementedError(
                        "selector_head is implemented for 3D only. No 2D wrapper."
                    )
                if cartesian:
                    from .selector_network import SelectorNetCartesian3D

                    return SelectorNetCartesian3D(
                        physical_size=size,
                        n_branches=config.selector_branches,
                        exit_time_features=config.exit_time_features,
                        **kwargs,
                    )
                from .selector_network import SelectorNet3D

                return SelectorNet3D(
                    physical_size=size,
                    n_branches=config.selector_branches,
                    fourier_angular=(ff.angular if ff is not None else False),
                    exit_time_features=config.exit_time_features,
                    **kwargs,
                )
            if config.domain_decomp is not None:
                if cartesian or dim != 3:
                    raise NotImplementedError(
                        "domain_decomp is implemented for 3D-spherical only. "
                        "Cartesian/2D wrappers are a follow-on."
                    )
                from .domain_decomp_network import DomainDecompNet3D

                return DomainDecompNet3D(
                    physical_size=size,
                    n_sub=config.domain_decomp,
                    overlap=config.domain_decomp_overlap,
                    fourier_angular=(ff.angular if ff is not None else False),
                    **kwargs,
                )
            if config.exit_time_features and dim != 3:
                raise NotImplementedError(
                    "exit_time_features is a 3D lever (Cartesian + spherical). "
                    "No 2D wrapper."
                )
            if cartesian:
                # No angular Fourier: d is already a smooth Cartesian vector.
                return FCMLPNetCartesian3D(
                    physical_size=size,
                    exit_time_features=config.exit_time_features,
                    exit_time_sign=config.exit_time_sign,
                    conditioning_dim=config.conditioning_dim,
                    field_encoder=config.field_encoder,
                    encoder_latent_dim=config.encoder_latent_dim,
                    encoder_hidden=config.encoder_hidden,
                    encoder_depth=config.encoder_depth,
                    **kwargs,
                )
            if dim == 3:
                # Angular Fourier is a 3D-spherical-only lever.
                return FCMLPNet3D(
                    physical_size=size,
                    fourier_angular=(ff.angular if ff is not None else False),
                    exit_time_features=config.exit_time_features,
                    exit_time_sign=config.exit_time_sign,
                    conditioning_dim=config.conditioning_dim,
                    field_encoder=config.field_encoder,
                    encoder_latent_dim=config.encoder_latent_dim,
                    encoder_hidden=config.encoder_hidden,
                    encoder_depth=config.encoder_depth,
                    **kwargs,
                )
            return FCMLPNet2D(physical_size=size, **kwargs)

        if isinstance(config, SIRENConfiguration):
            kwargs = dict(
                hidden_layers=config.hidden_layers,
                hidden_neurons=config.hidden_neurons,
                omega_0=config.omega_0,
                arch_name=config.arch_name,
                unit_norm_output=config.unit_norm_output,
                factored_eikonal=config.factored_eikonal,
            )
            if cartesian:
                return SIRENNetCartesian3D(physical_size=size, **kwargs)
            if dim == 3:
                return SIRENNet3D(physical_size=size, **kwargs)
            return SIRENNet2D(physical_size=size, **kwargs)

        raise ValueError(
            f"Unsupported network configuration type: "
            f"{type(config).__name__} (discriminator name={config.name!r})"
        )
