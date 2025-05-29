#ifndef WIRE_DATA_INIT_HPP
#define WIRE_DATA_INIT_HPP

#include <vector>
#include <random>
#include <cmath>
#include "experiments/run_params.hpp"

namespace wire {

struct wire_random_init {
    static std::vector<wire_cell_state> init(cellib::run::run_params& params) {
        // Initialize all cells to empty
        std::vector<wire_cell_state> initial_state(params.x_size * params.y_size, wire_cell_state::empty);
        
        // Calculate grid center
        const int center_x = params.x_size / 2;
        const int center_y = params.y_size / 2;
        
        // Create a random number generator with the provided seed if available
        std::mt19937 gen;
        gen.seed(static_cast<unsigned int>(params.seed));

        // Configuration: distance between consecutive loops
        const int loop_spacing = 4;
        
        // Vector to store conductor positions for placing electrons
        std::vector<std::pair<int, int>> conductor_positions;
        
        // Draw concentric rectangular loops
        int max_size = std::max(center_x, center_y);
        for (int size = loop_spacing; size <= max_size; size += loop_spacing) {
            // Calculate rectangle dimensions
            int half_width = size;
            int half_height = size / 2; // Make rectangles with 2:1 ratio
            
            // Ensure we don't go out of bounds
            if (center_x + half_width >= (int)(params.x_size) || center_x - half_width < 0 ||
                center_y + half_height >= (int)(params.y_size) || center_y - half_height < 0) {
                break;
            }
            
            // Draw the four sides of the rectangle
            // Top side
            for (int x = center_x - half_width; x <= center_x + half_width; x++) {
                int idx = (center_y - half_height) * params.x_size + x;
                initial_state[idx] = wire_cell_state::conductor;
                conductor_positions.push_back({x, center_y - half_height});
            }
            
            // Bottom side
            for (int x = center_x - half_width; x <= center_x + half_width; x++) {
                int idx = (center_y + half_height) * params.x_size + x;
                initial_state[idx] = wire_cell_state::conductor;
                conductor_positions.push_back({x, center_y + half_height});
            }
            
            // Left side (excluding corners, which are already drawn)
            for (int y = center_y - half_height + 1; y < center_y + half_height; y++) {
                int idx = y * params.x_size + (center_x - half_width);
                initial_state[idx] = wire_cell_state::conductor;
                conductor_positions.push_back({center_x - half_width, y});
            }
            
            // Right side (excluding corners, which are already drawn)
            for (int y = center_y - half_height + 1; y < center_y + half_height; y++) {
                int idx = y * params.x_size + (center_x + half_width);
                initial_state[idx] = wire_cell_state::conductor;
                conductor_positions.push_back({center_x + half_width, y});
            }
        }
        
        // Add random diagonal connections
        const int num_diagonals = 10; // Number of random diagonals
        
        std::uniform_int_distribution<int> x_dist(0, params.x_size - 1);
        std::uniform_int_distribution<int> y_dist(0, params.y_size - 1);
        std::uniform_int_distribution<int> length_dist(10, 30);
        
        for (int i = 0; i < num_diagonals; i++) {
            // Pick random start and end points
            int x1 = x_dist(gen);
            int y1 = y_dist(gen);
            
            // Choose a random direction and length
            int dx = (gen() % 3) - 1; // -1, 0 or 1
            int dy = (gen() % 3) - 1; // -1, 0 or 1
            
            // Ensure we're not drawing a zero-length line
            if (dx == 0 && dy == 0) dx = 1;
            
            int length = length_dist(gen);
            
            // Draw the diagonal line
            for (int j = 0; j < length; j++) {
                int x = x1 + j * dx;
                int y = y1 + j * dy;
                
                if (x >= 0 && x < static_cast<int>(params.x_size) && 
                    y >= 0 && y < static_cast<int>(params.y_size)) {
                    int idx = y * params.x_size + x;
                    initial_state[idx] = wire_cell_state::conductor;
                    conductor_positions.push_back({x, y});
                }
            }
        }
        
        // Add random circles
        const int num_circles = 5; // Number of random circles
        
        std::uniform_int_distribution<int> radius_dist(5, 15);
        
        for (int i = 0; i < num_circles; i++) {
            // Pick random center and radius
            int cx = x_dist(gen);
            int cy = y_dist(gen);
            int radius = radius_dist(gen);
            
            // Draw the circle
            for (int angle = 0; angle < 360; angle += 5) {
                // Convert angle to radians
                double rad = angle * M_PI / 180.0;
                
                // Calculate point on circle
                int x = cx + static_cast<int>(radius * cos(rad));
                int y = cy + static_cast<int>(radius * sin(rad));
                
                if (x >= 0 && x < static_cast<int>(params.x_size) && 
                    y >= 0 && y < static_cast<int>(params.y_size)) {
                    int idx = y * params.x_size + x;
                    initial_state[idx] = wire_cell_state::conductor;
                    conductor_positions.push_back({x, y});
                }
            }
        }
        
        // Add crossover bridges (extra feature - creates logical connections without short circuits)
        const int num_bridges = 8;
        
        for (int i = 0; i < num_bridges; i++) {
            int x = x_dist(gen);
            int y = y_dist(gen);
            
            // Make sure we're not too close to the edge
            if (x > 5 && x < static_cast<int>(params.x_size - 5) && 
                y > 5 && y < static_cast<int>(params.y_size - 5)) {
                
                // Create horizontal wire segment
                for (int dx = -4; dx <= 4; dx++) {
                    int idx = y * params.x_size + (x + dx);
                    initial_state[idx] = wire_cell_state::conductor;
                    conductor_positions.push_back({x + dx, y});
                }
                
                // Create vertical wire segment (with gaps for the crossover)
                for (int dy = -4; dy <= 4; dy++) {
                    if (dy != 0) { // Skip the intersection point
                        int idx = (y + dy) * params.x_size + x;
                        initial_state[idx] = wire_cell_state::conductor;
                        conductor_positions.push_back({x, y + dy});
                    }
                }
            }
        }
        
        // Place electron charges (heads followed by tails) at random positions
        if (!conductor_positions.empty()) {
            // Shuffle the conductor positions for random placement
            std::shuffle(conductor_positions.begin(), conductor_positions.end(), gen);
            
            // Place an electron head-tail pair every N conductor cells
            const int electron_spacing = 20; // Adjust for density of electrons
            
            for (size_t i = 0; i < conductor_positions.size(); i += electron_spacing) {
                if (i >= conductor_positions.size()) break;
                
                // Place an electron head
                auto [x, y] = conductor_positions[i];
                int idx = y * params.x_size + x;
                initial_state[idx] = wire_cell_state::electron_head;
                
                // Try to place an electron tail right after it in a valid neighbor
                bool tail_placed = false;
                
                // Define possible neighbor positions (4 directions)
                std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
                std::shuffle(directions.begin(), directions.end(), gen);
                
                for (auto [dx, dy] : directions) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    // Check bounds
                    if (nx >= 0 && nx < static_cast<int>(params.x_size) && 
                        ny >= 0 && ny < static_cast<int>(params.y_size)) {
                        
                        int neighbor_idx = ny * params.x_size + nx;
                        
                        // If neighbor is a conductor, make it a tail
                        if (initial_state[neighbor_idx] == wire_cell_state::conductor) {
                            initial_state[neighbor_idx] = wire_cell_state::electron_tail;
                            tail_placed = true;
                            break;
                        }
                    }
                }
                
                // If we couldn't place a tail, revert the head to a conductor
                if (!tail_placed) {
                    initial_state[idx] = wire_cell_state::conductor;
                }
            }
        }
        
        return initial_state;
    }
};

} // namespace wire

#endif // WIRE_DATA_INIT_HPP
