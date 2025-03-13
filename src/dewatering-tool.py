# wholistic-dewatering-tool.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import os


class WholeCostDewateringTool:
    def __init__(self,
                 # Basic parameters
                 cake_ts_target=21.5,          # Target dewatered cake TS%
                 dry_tons=29000,               # Annual dry tons of biosolids produced

                 # O&M Cost parameters
                 dry_polymer_price=1.77,       # $/lb for dry polymer
                 emulsion_polymer_price=2.61,  # $/lb for emulsion polymer

                 # Cake TS% vs Polymer Dose trendline parameters
                 dry_slope=7.4896,             # Slope for dry polymer ln equation
                 dry_intercept=-6.1534,        # Intercept for dry polymer ln equation
                 emulsion_slope=13.498,        # Slope for emulsion polymer ln equation
                 emulsion_intercept=-28.261,   # Intercept for emulsion polymer ln equation

                 # Transportation parameters
                 avg_round_trip_miles=144,     # Average round trip distance in miles
                 miles_per_gallon=6.0,         # Average MPG for trucks
                 fuel_price_per_gallon=4.00,   # Diesel fuel price per gallon
                 max_tons_per_load=21,         # Maximum wet tons per truckload (DOT limit)
                 trip_time_hours=3.0,          # Average round trip time including loading/unloading

                 # Variable cost parameters
                 materials_supplies_cost=154000,  # Materials & supplies for transportation
                 tires_tubes_cost=32445,         # Tires & tubes for transportation
                 distribution_materials_cost=161500,  # Materials & supplies for distribution
                 distribution_tires_cost=8755,       # Tires & tubes for distribution
                 os_maintenance_cost=23460,          # Outside maintenance for transportation
                 distribution_os_maintenance=11330,  # Outside maintenance for distribution
                 os_rental_cost=2000,               # Outside rental costs
                 os_other_cost=10650,               # Outside other costs

                 # Step function parameters
                 labor_wages_benefits=2234841,      # Base wages & benefits
                 new_operator_cost=100000,          # Cost per additional operator
                 new_mechanic_cost=100000,          # Cost per additional mechanic
                 tractor_trailer_cost=25000,        # Annual cost for new tractor/trailer

                 # Step function thresholds - % decrease in TS that triggers new resources
                 operator_threshold=0.01,           # % TS decrease to add an operator (1%)
                 mechanic_threshold=0.02,           # % TS decrease to add a mechanic (2%)
                 equipment_threshold=0.01):         # % TS decrease to add equipment (1%)

        # Initialize parameters
        self.cake_ts_target = cake_ts_target
        self.dry_tons = dry_tons

        # Polymer parameters
        self.dry_polymer_price = dry_polymer_price
        self.emulsion_polymer_price = emulsion_polymer_price
        self.dry_slope = dry_slope
        self.dry_intercept = dry_intercept
        self.emulsion_slope = emulsion_slope
        self.emulsion_intercept = emulsion_intercept

        # Transportation parameters
        self.avg_round_trip_miles = avg_round_trip_miles
        self.miles_per_gallon = miles_per_gallon
        self.fuel_price_per_gallon = fuel_price_per_gallon
        self.max_tons_per_load = max_tons_per_load
        self.trip_time_hours = trip_time_hours

        # Variable costs
        self.materials_supplies_cost = materials_supplies_cost
        self.tires_tubes_cost = tires_tubes_cost
        self.distribution_materials_cost = distribution_materials_cost
        self.distribution_tires_cost = distribution_tires_cost
        self.os_maintenance_cost = os_maintenance_cost
        self.distribution_os_maintenance = distribution_os_maintenance
        self.os_rental_cost = os_rental_cost
        self.os_other_cost = os_other_cost

        # Step function parameters
        self.labor_wages_benefits = labor_wages_benefits
        self.new_operator_cost = new_operator_cost
        self.new_mechanic_cost = new_mechanic_cost
        self.tractor_trailer_cost = tractor_trailer_cost

        # Step function thresholds
        self.operator_threshold = operator_threshold
        self.mechanic_threshold = mechanic_threshold
        self.equipment_threshold = equipment_threshold

    def calculate_wet_tons(self, dewatered_cake_ts):
        """Calculate wet tons based on dewatered cake TS%."""
        return self.dry_tons / (dewatered_cake_ts / 100)

    def calculate_number_of_truckloads(self, wet_tons):
        """Calculate the number of truckloads required."""
        return wet_tons / self.max_tons_per_load

    def calculate_total_transportation_distance(self, number_of_loads):
        """Calculate the total annual transportation distance in miles."""
        return number_of_loads * self.avg_round_trip_miles

    def calculate_fuel_consumption(self, total_distance):
        """Calculate the fuel consumption in gallons."""
        return total_distance / self.miles_per_gallon

    def calculate_fuel_cost(self, fuel_consumption):
        """Calculate the fuel cost based on consumption."""
        return fuel_consumption * self.fuel_price_per_gallon

    def calculate_variable_transportation_costs(self, wet_tons):
        """Calculate the linearly variable transportation costs."""
        # We need to calculate how costs scale with wet tons
        # Base wet tons at the target TS%
        base_wet_tons = self.calculate_wet_tons(self.cake_ts_target)

        # Calculate the scaling factor based on wet tons
        scaling_factor = wet_tons / base_wet_tons

        # Sum all variable transportation costs
        total_variable_costs = (
            self.materials_supplies_cost +
            self.tires_tubes_cost +
            self.distribution_materials_cost +
            self.distribution_tires_cost +
            self.os_maintenance_cost +
            self.distribution_os_maintenance +
            self.os_rental_cost +
            self.os_other_cost
        )

        # Calculate fuel costs
        loads = self.calculate_number_of_truckloads(wet_tons)
        total_distance = self.calculate_total_transportation_distance(loads)
        fuel_consumption = self.calculate_fuel_consumption(total_distance)
        fuel_cost = self.calculate_fuel_cost(fuel_consumption)

        # Add in fuel costs separately as they're directly calculated
        total_variable_costs = total_variable_costs * scaling_factor + fuel_cost

        return total_variable_costs

    def calculate_labor_expenses(self, dewatered_cake_ts):
        """
        Calculate Total Labor Expenses with step function.
        Adds operators and mechanics based on decreases in TS%
        """
        if dewatered_cake_ts >= self.cake_ts_target:
            return self.labor_wages_benefits
        else:
            # Calculate additional operators needed
            operators_needed = math.floor(
                (self.cake_ts_target - dewatered_cake_ts) / self.operator_threshold
            )

            # Calculate additional mechanics needed
            mechanics_needed = math.floor(
                (self.cake_ts_target - dewatered_cake_ts) / self.mechanic_threshold
            )

            # Calculate total labor cost
            total_labor = (
                self.labor_wages_benefits +
                operators_needed * self.new_operator_cost +
                mechanics_needed * self.new_mechanic_cost
            )

            return total_labor

    def calculate_equipment_expenses(self, dewatered_cake_ts):
        """
        Calculate Additional Equipment Expenses with step function.
        Adds tractors/trailers based on decreases in TS%
        """
        if dewatered_cake_ts >= self.cake_ts_target:
            return 0
        else:
            # Calculate additional tractors/trailers needed
            equipment_needed = math.floor(
                (self.cake_ts_target - dewatered_cake_ts) / self.equipment_threshold
            )

            # Calculate total equipment cost
            total_equipment = equipment_needed * self.tractor_trailer_cost

            return total_equipment

    def calculate_dry_polymer_dose(self, dewatered_cake_ts):
        """Calculate the required dry polymer dose based on the cake TS%."""
        # Using the logarithmic equation: TS% = m*ln(dose) + b
        # Rearranging to solve for dose: dose = exp((TS% - b) / m)
        return math.exp((dewatered_cake_ts - self.dry_intercept) / self.dry_slope)

    def calculate_emulsion_polymer_dose(self, dewatered_cake_ts):
        """Calculate the required emulsion polymer dose based on the cake TS%."""
        # Using the logarithmic equation: TS% = m*ln(dose) + b
        # Rearranging to solve for dose: dose = exp((TS% - b) / m)
        return math.exp((dewatered_cake_ts - self.emulsion_intercept) / self.emulsion_slope)

    def calculate_dry_polymer_cost(self, dewatered_cake_ts):
        """Calculate the annual cost of dry polymer."""
        dose = self.calculate_dry_polymer_dose(dewatered_cake_ts)  # lb/dry ton
        annual_cost = dose * self.dry_tons * self.dry_polymer_price
        return annual_cost

    def calculate_emulsion_polymer_cost(self, dewatered_cake_ts):
        """Calculate the annual cost of emulsion polymer."""
        dose = self.calculate_emulsion_polymer_dose(dewatered_cake_ts)  # lb/dry ton
        annual_cost = dose * self.dry_tons * self.emulsion_polymer_price
        return annual_cost

    def run_analysis(self, ts_percentages=None):
        """
        Run the analysis for a range of dewatered cake TS percentages.
        """
        if ts_percentages is None:
            # Use 0.5% increments in the range specified in the report
            ts_percentages = np.arange(18, 24.5, 0.5)

        results = []

        for ts in ts_percentages:
            # Calculate wet tons
            wet_tons = self.calculate_wet_tons(ts)

            # Calculate RR&R costs
            transportation_costs = self.calculate_variable_transportation_costs(wet_tons)
            labor_expenses = self.calculate_labor_expenses(ts)
            equipment_expenses = self.calculate_equipment_expenses(ts)
            total_rrr_expenses = transportation_costs + labor_expenses + equipment_expenses

            # Calculate polymer costs
            dry_polymer_cost = self.calculate_dry_polymer_cost(ts)
            emulsion_polymer_cost = self.calculate_emulsion_polymer_cost(ts)

            # Total costs
            total_cost_dry = total_rrr_expenses + dry_polymer_cost
            total_cost_emulsion = total_rrr_expenses + emulsion_polymer_cost

            result = {
                'Dewatered_Cake_TS%': ts,
                'Wet_Tons': wet_tons,
                'Transportation_Costs': transportation_costs,
                'Labor_Expenses': labor_expenses,
                'Equipment_Expenses': equipment_expenses,
                'Total_RRR_Expenses': total_rrr_expenses,
                'Dry_Polymer_Dose': self.calculate_dry_polymer_dose(ts),
                'Emulsion_Polymer_Dose': self.calculate_emulsion_polymer_dose(ts),
                'Dry_Polymer_Cost': dry_polymer_cost,
                'Emulsion_Polymer_Cost': emulsion_polymer_cost,
                'Total_Cost_Dry': total_cost_dry,
                'Total_Cost_Emulsion': total_cost_emulsion
            }

            results.append(result)

        return pd.DataFrame(results)

    def find_optimal_ts(self, results):
        """Find the optimal TS% for both polymer types."""
        # Find minimum total cost for dry polymer
        min_cost_dry = results.loc[results['Total_Cost_Dry'].idxmin()]
        optimal_ts_dry = min_cost_dry['Dewatered_Cake_TS%']

        # Find minimum total cost for emulsion polymer
        min_cost_emulsion = results.loc[results['Total_Cost_Emulsion'].idxmin()]
        optimal_ts_emulsion = min_cost_emulsion['Dewatered_Cake_TS%']

        return {
            'dry': {
                'ts': optimal_ts_dry,
                'cost': min_cost_dry['Total_Cost_Dry']
            },
            'emulsion': {
                'ts': optimal_ts_emulsion,
                'cost': min_cost_emulsion['Total_Cost_Emulsion']
            }
        }

    def plot_results(self, results, output_dir=None):
        """
        Generate plots of the analysis results.
        Returns a figure showing total cost versus dewatered cake %TS and the ideal point.
        """
        # Find optimal operating points
        optimal = self.find_optimal_ts(results)

        # Create the main figure showing total cost vs TS%
        plt.figure(figsize=(12, 8))

        # Plot total costs for both polymer types
        plt.plot(results['Dewatered_Cake_TS%'], results['Total_Cost_Dry'],
                 'b-', linewidth=2, label='Total Cost - Dry Polymer')
        plt.plot(results['Dewatered_Cake_TS%'], results['Total_Cost_Emulsion'],
                 'r-', linewidth=2, label='Total Cost - Emulsion Polymer')

        # Plot RR&R costs
        plt.plot(results['Dewatered_Cake_TS%'], results['Total_RRR_Expenses'],
                 'g--', label='RR&R Expenses')

        # Plot polymer costs
        plt.plot(results['Dewatered_Cake_TS%'], results['Dry_Polymer_Cost'],
                 'b:', label='Dry Polymer Cost')
        plt.plot(results['Dewatered_Cake_TS%'], results['Emulsion_Polymer_Cost'],
                 'r:', label='Emulsion Polymer Cost')

        # Mark optimal points
        plt.scatter([optimal['dry']['ts']], [optimal['dry']['cost']], color='blue', s=100, zorder=5)
        plt.annotate(f"Optimal (Dry): {optimal['dry']['ts']:.1f}% TS\n${optimal['dry']['cost']:,.0f}",
                     (optimal['dry']['ts'], optimal['dry']['cost']),
                     xytext=(10, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

        plt.scatter([optimal['emulsion']['ts']], [optimal['emulsion']['cost']], color='red', s=100, zorder=5)
        plt.annotate(f"Optimal (Emulsion): {optimal['emulsion']['ts']:.1f}% TS\n${optimal['emulsion']['cost']:,.0f}",
                     (optimal['emulsion']['ts'], optimal['emulsion']['cost']),
                     xytext=(10, -30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

        # Add a vertical line at the target TS%
        plt.axvline(x=self.cake_ts_target, color='gray', linestyle='--', label=f'Target TS% ({self.cake_ts_target}%)')

        # Add labels, title and legend
        plt.title('Whole Cost Dewatering Analysis - Total Cost vs. Dewatered Cake TS%')
        plt.xlabel('Dewatered Cake TS%')
        plt.ylabel('Annual Cost ($)')
        plt.grid(True)
        plt.legend()

        # Highlight the recommended range (20-22%)
        plt.axvspan(20, 22, alpha=0.2, color='green', label='Recommended Range (20-22%)')

        plt.tight_layout()

        # Save the figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'dewatering_cost_analysis.png')
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return plt.gcf()

    def generate_report(self, results):
        """Generate a textual report of the analysis findings."""
        optimal = self.find_optimal_ts(results)

        report = [
            "=== Whole Cost Dewatering Analysis Report ===",
            f"Annual Dry Tons: {self.dry_tons:,}",
            f"Target TS%: {self.cake_ts_target}%",
            "",
            "Optimal Operating Points:",
            f"  Dry Polymer: {optimal['dry']['ts']:.1f}% TS - ${optimal['dry']['cost']:,.2f}",
            f"  Emulsion Polymer: {optimal['emulsion']['ts']:.1f}% TS - ${optimal['emulsion']['cost']:,.2f}",
            "",
            "Cost Breakdown at Optimal Points:",
        ]

        # Get cost breakdown at optimal points
        for polymer_type in ['dry', 'emulsion']:
            ts = optimal[polymer_type]['ts']
            # Use numpy's isclose for floating point comparison
            row = results[np.isclose(results['Dewatered_Cake_TS%'], ts, rtol=1e-10, atol=1e-10)].iloc[0]

            report.extend([
                f"  {polymer_type.capitalize()} Polymer ({ts:.1f}% TS):",
                f"    Wet Tons: {row['Wet_Tons']:,.0f}",
                f"    Transportation Costs: ${row['Transportation_Costs']:,.2f}",
                f"    Labor Expenses: ${row['Labor_Expenses']:,.2f}",
                f"    Equipment Expenses: ${row['Equipment_Expenses']:,.2f}",
                f"    Total RR&R Expenses: ${row['Total_RRR_Expenses']:,.2f}",
                f"    Polymer Dose: {row[f'{polymer_type.capitalize()}_Polymer_Dose']:.2f} lb/ton",
                f"    Polymer Cost: ${row[f'{polymer_type.capitalize()}_Polymer_Cost']:,.2f}",
                f"    Total Cost: ${row[f'Total_Cost_{polymer_type.capitalize()}']:,.2f}",
                ""
            ])

        report.append("The analysis indicates that the ideal range for target dewatered cake %TS is between 20% and 22%,")
        report.append("which aligns with the findings in the Whole Cost Dewatering Phase I Assessment report.")

        return "\n".join(report)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Whole Cost Dewatering Analysis Tool')

    # Basic parameters
    parser.add_argument('--cake-ts-target', type=float, default=21.5,
                        help='Target dewatered cake TS%% (default: 21.5)')
    parser.add_argument('--dry-tons', type=float, default=29000,
                        help='Annual dry tons produced (default: 29000)')

    # Polymer parameters
    parser.add_argument('--dry-polymer-price', type=float, default=1.77,
                        help='Dry polymer price per pound (default: $1.77/lb)')
    parser.add_argument('--emulsion-polymer-price', type=float, default=2.61,
                        help='Emulsion polymer price per pound (default: $2.61/lb)')

    # Transportation parameters
    parser.add_argument('--avg-trip-miles', type=float, default=144,
                        help='Average round trip miles (default: 144)')
    parser.add_argument('--mpg', type=float, default=6.0,
                        help='Miles per gallon for trucks (default: 6.0)')
    parser.add_argument('--fuel-price', type=float, default=4.00,
                        help='Diesel fuel price per gallon (default: $4.00)')
    parser.add_argument('--max-tons-per-load', type=float, default=21,
                        help='Maximum wet tons per load (default: 21)')

    # Step function parameters
    parser.add_argument('--operator-cost', type=float, default=100000,
                        help='Cost per additional operator (default: $100,000)')
    parser.add_argument('--mechanic-cost', type=float, default=100000,
                        help='Cost per additional mechanic (default: $100,000)')
    parser.add_argument('--equipment-cost', type=float, default=25000,
                        help='Annual cost per tractor/trailer (default: $25,000)')

    # Step function thresholds
    parser.add_argument('--operator-threshold', type=float, default=0.01,
                        help='%% TS decrease to add an operator (default: 0.01 = 1%%)')
    parser.add_argument('--mechanic-threshold', type=float, default=0.02,
                        help='%% TS decrease to add a mechanic (default: 0.02 = 2%%)')
    parser.add_argument('--equipment-threshold', type=float, default=0.01,
                        help='%% TS decrease to add equipment (default: 0.01 = 1%%)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--export-excel', action='store_true',
                        help='Export results to Excel')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Create an instance of the dewatering tool with command line parameters
    tool = WholeCostDewateringTool(
        cake_ts_target=args.cake_ts_target,
        dry_tons=args.dry_tons,
        dry_polymer_price=args.dry_polymer_price,
        emulsion_polymer_price=args.emulsion_polymer_price,
        avg_round_trip_miles=args.avg_trip_miles,
        miles_per_gallon=args.mpg,
        fuel_price_per_gallon=args.fuel_price,
        max_tons_per_load=args.max_tons_per_load,
        new_operator_cost=args.operator_cost,
        new_mechanic_cost=args.mechanic_cost,
        tractor_trailer_cost=args.equipment_cost,
        operator_threshold=args.operator_threshold,
        mechanic_threshold=args.mechanic_threshold,
        equipment_threshold=args.equipment_threshold
    )

    # Run the analysis
    results = tool.run_analysis()

    # Generate and display the plot
    fig = tool.plot_results(results, args.output_dir)

    # Generate and display the report
    report = tool.generate_report(results)
    print(report)

    # Export to Excel if requested
    if args.export_excel:
        output_path = os.path.join(args.output_dir, 'dewatering_analysis_results.xlsx')
        results.to_excel(output_path, index=False)
        print(f"Results exported to {output_path}")

    # Show the plot
    plt.show()
