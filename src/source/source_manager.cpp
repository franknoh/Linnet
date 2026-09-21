#include "linnet/source/source_manager.hpp"

#include "linnet/source/utf8.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iterator>
#include <utility>

namespace linnet {

std::expected<FileId, std::string> SourceManager::add_file(std::string path, std::string contents) {
    if (contents.size() > max_file_size) {
        return std::unexpected(path + ": file is too large");
    }
    if (files_.size() >= invalid_file_id) {
        return std::unexpected(path + ": too many source files");
    }

    File file{std::move(path), std::move(contents), {0}};
    for (std::size_t i = 0; i < file.contents.size(); ++i) {
        if (file.contents[i] == '\n') {
            file.line_starts.push_back(static_cast<std::uint32_t>(i + 1));
        }
    }

    files_.push_back(std::move(file));
    return static_cast<FileId>(files_.size() - 1);
}

std::expected<FileId, std::string> SourceManager::load_file(const std::filesystem::path& path) {
    const std::string display = path.generic_string();

    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        return std::unexpected(display + ": not a readable file");
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::unexpected(display + ": cannot open file");
    }
    std::string contents{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        return std::unexpected(display + ": read error");
    }
    return add_file(display, std::move(contents));
}

const SourceManager::File& SourceManager::file(FileId id) const {
    assert(id < files_.size());
    return files_[id];
}

std::string_view SourceManager::path(FileId id) const {
    return file(id).path;
}

std::string_view SourceManager::contents(FileId id) const {
    return file(id).contents;
}

std::uint32_t SourceManager::size(FileId id) const {
    return static_cast<std::uint32_t>(file(id).contents.size());
}

std::string_view SourceManager::text(SourceSpan span) const {
    const std::string_view all = contents(span.file);
    assert(span.begin <= span.end && span.end <= all.size());
    return all.substr(span.begin, span.end - span.begin);
}

std::uint32_t SourceManager::line_count(FileId id) const {
    return static_cast<std::uint32_t>(file(id).line_starts.size());
}

std::uint32_t SourceManager::line_index(const File& file, std::uint32_t offset) {
    const auto next = std::upper_bound(file.line_starts.begin(), file.line_starts.end(), offset);
    return static_cast<std::uint32_t>(std::distance(file.line_starts.begin(), next) - 1);
}

std::uint32_t SourceManager::line_end(const File& file, std::uint32_t line_index) {
    const std::string& contents = file.contents;
    if (line_index + 1 >= file.line_starts.size()) {
        return static_cast<std::uint32_t>(contents.size());
    }
    std::uint32_t end = file.line_starts[line_index + 1] - 1; // at the '\n'
    if (end > file.line_starts[line_index] && contents[end - 1] == '\r') {
        --end;
    }
    return end;
}

std::uint32_t SourceManager::line_start(FileId id, std::uint32_t line) const {
    const File& source = file(id);
    assert(line >= 1 && line <= source.line_starts.size());
    return source.line_starts[line - 1];
}

std::string_view SourceManager::line_text(FileId id, std::uint32_t line) const {
    const File& source = file(id);
    assert(line >= 1 && line <= source.line_starts.size());
    const std::uint32_t begin = source.line_starts[line - 1];
    const std::uint32_t end = line_end(source, line - 1);
    return std::string_view(source.contents).substr(begin, end - begin);
}

LineColumn SourceManager::line_column(FileId id, std::uint32_t offset) const {
    const File& source = file(id);
    offset = std::min(offset, static_cast<std::uint32_t>(source.contents.size()));
    const std::uint32_t index = line_index(source, offset);
    return {index + 1, offset - source.line_starts[index] + 1};
}

Utf16Position SourceManager::utf16_position(FileId id, std::uint32_t offset) const {
    const File& source = file(id);
    offset = std::min(offset, static_cast<std::uint32_t>(source.contents.size()));
    const std::uint32_t index = line_index(source, offset);

    std::uint32_t character = 0;
    std::size_t cursor = source.line_starts[index];
    while (cursor < offset) {
        const DecodedCodePoint decoded = decode_utf8(source.contents, cursor);
        if (cursor + decoded.length > offset) {
            break; // offset points inside a code point; report its start
        }
        character += utf16_length(decoded.value);
        cursor += decoded.length;
    }
    return {index, character};
}

std::uint32_t SourceManager::offset_from_utf16(FileId id, Utf16Position position) const {
    const File& source = file(id);
    if (position.line >= source.line_starts.size()) {
        return static_cast<std::uint32_t>(source.contents.size());
    }

    const std::uint32_t end = line_end(source, position.line);
    std::uint32_t cursor = source.line_starts[position.line];
    std::uint32_t character = 0;
    while (cursor < end) {
        const DecodedCodePoint decoded = decode_utf8(source.contents, cursor);
        const std::uint32_t units = utf16_length(decoded.value);
        if (character + units > position.character) {
            break;
        }
        character += units;
        cursor += decoded.length;
    }
    return cursor;
}

} // namespace linnet
