# Northwind Supply Network — Dependencies

The Northwind supply network connects raw-material suppliers to component plants and onward to distribution centers.

Chain of dependency:
1. Crude polymers are supplied by PetroKem Co. to the Ohio plant.
2. The Ohio plant manufactures adhesives, which are shipped to the Memphis distribution center.
3. The Memphis distribution center ships finished goods to all retail distributors in the eastern United States.
4. The Ohio plant is itself the sole supplier of sealant base to the Monterrey, Mexico plant.

Therefore, any disruption at PetroKem Co. propagates first to the Ohio plant, then to the Memphis center, and finally to eastern US retail distributors. The Monterrey plant is downstream of Ohio, so it depends indirectly on PetroKem.

The Ohio plant relies on a single upstream supplier (PetroKem) for its crude polymers, making it a single point of dependency.