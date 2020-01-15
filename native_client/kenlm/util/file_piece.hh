#ifndef UTIL_FILE_PIECE_H
#define UTIL_FILE_PIECE_H

#include "util/ersatz_progress.hh"
#include "util/exception.hh"
#include "util/file.hh"
#include "util/mmap.hh"
#include "util/read_compressed.hh"
#include "util/spaces.hh"
#include "util/string_piece.hh"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <cassert>
#include <stdint.h>

namespace util {

class ParseNumberException : public Exception {
  public:
    explicit ParseNumberException(StringPiece value) throw();
    ~ParseNumberException() throw() {}
};

class FilePiece;

// Input Iterator over lines.  This allows
//   for (StringPiece l : FilePiece("file"))
// in C++11.
// NB: not multipass.
class LineIterator {
  public:
    LineIterator() : backing_(NULL) {}

    explicit LineIterator(FilePiece &f, char delim = '\n') : backing_(&f), delim_(delim) {
      ++*this;
    }

    LineIterator &operator++();

    bool operator==(const LineIterator &other) const {
      return backing_ == other.backing_;
    }

    bool operator!=(const LineIterator &other) const {
      return backing_ != other.backing_;
    }

    operator bool() const { return backing_ != NULL; }

    StringPiece operator*() const { return line_; }
    const StringPiece *operator->() const { return &line_; }

  private:
    FilePiece *backing_;
    StringPiece line_;
    char delim_;
};

// Memory backing the returned StringPiece may vanish on the next call.
class FilePiece {
  public:
    // 1 MB default.
    explicit FilePiece(const char *file, std::ostream *show_progress = NULL, std::size_t min_buffer = 1048576);
    // Takes ownership of fd.  name is used for messages.
    explicit FilePiece(int fd, const char *name = NULL, std::ostream *show_progress = NULL, std::size_t min_buffer = 1048576);

    /* Read from an istream.  Don't use this if you can avoid it.  Raw fd IO is
     * much faster.  But sometimes you just have an istream like Boost's HTTP
     * server and want to parse it the same way.
     * name is just used for messages and FileName().
     */
    explicit FilePiece(std::istream &stream, const char *name = NULL, std::size_t min_buffer = 1048576);

    LineIterator begin() {
      return LineIterator(*this);
    }

    LineIterator end() {
      return LineIterator();
    }

    char get() {
      if (position_ == position_end_) {
        Shift();
        if (at_end_) throw EndOfFileException();
      }
      return *(position_++);
    }

    // Leaves the delimiter, if any, to be returned by get().  Delimiters defined by isspace().
    StringPiece ReadDelimited(const bool *delim = kSpaces) {
      SkipSpaces(delim);
      return Consume(FindDelimiterOrEOF(delim));
    }

    /// Read word until the line or file ends.
    bool ReadWordSameLine(StringPiece &to, const bool *delim = kSpaces) {
      assert(delim[static_cast<unsigned char>('\n')]);
      // Skip non-enter spaces.
      for (; ; ++position_) {
        if (position_ == position_end_) {
          try {
            Shift();
          } catch (const util::EndOfFileException &) { return false; }
          // And break out at end of file.
          if (position_ == position_end_) return false;
        }
        if (!delim[static_cast<unsigned char>(*position_)]) break;
        if (*position_ == '\n') return false;
      }
      // We can't be at the end of file because there's at least one character open.
      to = Consume(FindDelimiterOrEOF(delim));
      return true;
    }

    /** Read a line of text from the file.
     *
     * Unlike ReadDelimited, this includes leading spaces and consumes the
     * delimiter.   It is similar to getline in that way.
     *
     * If strip_cr is true, any trailing carriate return (as would be found on
     * a file written on Windows) will be left out of the returned line.
     *
     * Throws EndOfFileException if the end of the file is encountered.  If the
     * file does not end in a newline, this could mean that the last line is
     * never read.
     */
    StringPiece ReadLine(char delim = '\n', bool strip_cr = true);

    /** Read a line of text from the file, or return false on EOF.
     *
     * This is like ReadLine, except it returns false where ReadLine throws
     * EndOfFileException.  Like ReadLine it may not read the last line in the
     * file if the file does not end in a newline.
     *
     * If strip_cr is true, any trailing carriate return (as would be found on
     * a file written on Windows) will be left out of the returned line.
     */
    bool ReadLineOrEOF(StringPiece &to, char delim = '\n', bool strip_cr = true);

    float ReadFloat();
    double ReadDouble();
    long int ReadLong();
    unsigned long int ReadULong();

    // Skip spaces defined by isspace.
    void SkipSpaces(const bool *delim = kSpaces) {
      assert(position_ <= position_end_);
      for (; ; ++position_) {
        if (position_ == position_end_) {
          Shift();
          // And break out at end of file.
          if (position_ == position_end_) return;
        }
        assert(position_ < position_end_);
        if (!delim[static_cast<unsigned char>(*position_)]) return;
      }
    }

    uint64_t Offset() const {
      return position_ - data_.begin() + mapped_offset_;
    }

    const std::string &FileName() const { return file_name_; }

    // Force a progress update.
    void UpdateProgress();

  private:
    void InitializeNoRead(const char *name, std::size_t min_buffer);
    // Calls InitializeNoRead, so don't call both.
    void Initialize(const char *name, std::ostream *show_progress, std::size_t min_buffer);

    template <class T> T ReadNumber();

    StringPiece Consume(const char *to) {
      assert(to >= position_);
      StringPiece ret(position_, to - position_);
      position_ = to;
      return ret;
    }

    const char *FindDelimiterOrEOF(const bool *delim = kSpaces);

    void Shift();
    // Backends to Shift().
    void MMapShift(uint64_t desired_begin);

    void TransitionToRead();
    void ReadShift();

    const char *position_, *last_space_, *position_end_;

    scoped_fd file_;
    const uint64_t total_size_;

    std::size_t default_map_size_;
    uint64_t mapped_offset_;

    // Order matters: file_ should always be destroyed after this.
    scoped_memory data_;

    bool at_end_;
    bool fallback_to_read_;

    ErsatzProgress progress_;

    std::string file_name_;

    ReadCompressed fell_back_;
};

} // namespace util

#endif // UTIL_FILE_PIECE_H
