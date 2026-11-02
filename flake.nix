{
  description = "Machine Learning flake";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.zst";
  };

  outputs = inputs: {
    devShells = builtins.mapAttrs (system: pkgs: {
	    default = pkgs.mkShell {
		packages = with pkgs; [
		    python314
		    uv
		    #python314Packages.pandas
		    python314Packages.numpy
		    python314Packages.livereload
		];

		shellHook = ''
		    source .venv/bin/activate 
		    '';
	    };
    }) inputs.nixpkgs.legacyPackages;
  };
}
